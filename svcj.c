#include "svcj.h"
#include <float.h>

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) if(fabs(returns[i]) < 1e-12) returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
}

void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS + IDX_CLOSE];
        double curr = ohlcv[i*N_COLS + IDX_CLOSE];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr / prev);
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1;     if(p->kappa > 50.0) p->kappa = 50.0;
    if(p->theta < 0.0001) p->theta = 0.0001; if(p->theta > 2.0) p->theta = 2.0; 
    if(p->sigma_v < 0.01) p->sigma_v = 0.01; if(p->sigma_v > 8.0) p->sigma_v = 8.0;
    if(p->rho > 0.99) p->rho = 0.99;       if(p->rho < -0.99) p->rho = -0.99;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001; if(p->lambda_j > 300.0) p->lambda_j = 300.0; 
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 20.0) p->sigma_v = sqrt(feller * 20.0);
}

void estimate_initial_params_smart(double* ohlcv, int n, SVCJParams* p) {
    double sum_gk = 0.0;
    for(int i=0; i<n; i++) {
        double O=ohlcv[i*N_COLS+0]; double H=ohlcv[i*N_COLS+1];
        double L=ohlcv[i*N_COLS+2]; double C=ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        double hl=log(H/L); double co=log(C/O);
        double val = 0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
        if(val>0) sum_gk+=val;
    }
    double rv_annual = (sum_gk/n) * 252.0;
    p->mu = 0.0; p->theta = rv_annual; if(p->theta < 0.0025) p->theta = 0.0025;
    p->kappa = 4.0; p->sigma_v = sqrt(p->theta) * 2.0; p->rho = -0.6;
    p->lambda_j = 0.1; p->mu_j = 0.0; p->sigma_j = sqrt(rv_annual/252.0) * 3.0; 
    check_constraints(p);
}

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob, double theta_anchor) {
    double ll = 0.0; double v = p->theta; double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;
        double y = returns[t] - (p->mu - 0.5 * v_pred) * DT;
        double robust_var_d = fmax(v_pred, 0.1 * p->theta) * DT; 
        double pdf_d = (1.0 / (sqrt(robust_var_d) * SQRT_2PI)) * exp(-0.5 * y*y / robust_var_d);
        double total_var_j = robust_var_d + var_j; double y_j = y - p->mu_j; 
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        double prob_prior = (p->lambda_j * DT > 0.9) ? 0.9 : p->lambda_j * DT;
        double den = pdf_j * prob_prior + pdf_d * (1.0 - prob_prior);
        if(den < 1e-15) den = 1e-15;
        double prob_posterior = (pdf_j * prob_prior) / den;
        double S = v_pred * DT + prob_posterior * var_j;
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        if(v < 1e-6) v = 1e-6; if(v > 5.0) v = 5.0;
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); 
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;
        ll += log(den); 
    }
    double complexity_penalty = -OCCAM_WEIGHT * p->lambda_j;
    double theta_penalty = -20.0 * pow(log(p->theta) - log(theta_anchor), 2);
    return ll + theta_penalty + complexity_penalty;
}

void grid_search_init(double* returns, int n, SVCJParams* p, double theta_anchor) {
    double lambdas[] = {0.05, 0.5, 5.0}; double vol_vols[] = {0.5, 2.0, 5.0};
    double best_score = -1e15; SVCJParams best_p = *p;
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) {
        SVCJParams temp = *p; temp.lambda_j = lambdas[i]; temp.sigma_v = vol_vols[j]; temp.theta = theta_anchor;
        check_constraints(&temp);
        double score = ukf_log_likelihood(returns, n, &temp, NULL, NULL, theta_anchor);
        if(score > best_score) { best_score = score; best_p = temp; }
    }
    *p = best_p;
}

void optimize_svcj(double* ohlcv, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params_smart(ohlcv, n, p);
    double theta_anchor = p->theta;
    int n_ret = n - 1;
    double* returns = (double*)malloc(n_ret * sizeof(double));
    if(!returns) return;
    compute_log_returns(ohlcv, n, returns);
    clean_returns(returns, n_ret);
    grid_search_init(returns, n_ret, p, theta_anchor);
    int n_dim = 5; double simplex[6][5]; double scores[6];
    double global_best = -1e15; SVCJParams final_p = *p;
    for(int r=0; r<2; r++) {
        if(r>0) { p->sigma_v *= 1.5; check_constraints(p); } 
        for(int i=0; i<=n_dim; i++) {
            SVCJParams temp = *p;
            if(i==1) temp.kappa *= 1.2; if(i==2) temp.theta *= 1.2; if(i==3) temp.sigma_v *= 1.2; 
            if(i==4) temp.rho -= 0.1; if(i==5) temp.lambda_j *= 1.5;
            check_constraints(&temp);
            simplex[i][0]=temp.kappa; simplex[i][1]=temp.theta; simplex[i][2]=temp.sigma_v;
            simplex[i][3]=temp.rho; simplex[i][4]=temp.lambda_j;
            scores[i] = ukf_log_likelihood(returns, n_ret, &temp, NULL, NULL, theta_anchor);
        }
        for(int iter=0; iter<NM_ITER; iter++) {
            int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
            for(int i=0; i<6; i++) for(int j=i+1; j<6; j++) if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
            double c[5]={0}; for(int i=0; i<5; i++) for(int k=0; k<5; k++) c[k]+=simplex[vs[i]][k]; for(int k=0; k<5; k++) c[k]/=5.0;
            double ref[5]; SVCJParams rp=*p; for(int k=0; k<5; k++) ref[k]=c[k]+1.0*(c[k]-simplex[vs[5]][k]); check_constraints(&rp);
            double r_score=ukf_log_likelihood(returns, n_ret, &rp, NULL, NULL, theta_anchor);
            if(r_score>scores[vs[0]]) {
                double exp[5]; SVCJParams ep=*p; for(int k=0; k<5; k++) exp[k]=c[k]+2.0*(c[k]-simplex[vs[5]][k]); check_constraints(&ep);
                double e_score=ukf_log_likelihood(returns, n_ret, &ep, NULL, NULL, theta_anchor);
                if(e_score>r_score) { for(int k=0; k<5; k++) simplex[vs[5]][k]=exp[k]; scores[vs[5]]=e_score; } else { for(int k=0; k<5; k++) simplex[vs[5]][k]=ref[k]; scores[vs[5]]=r_score; }
            } else if(r_score>scores[vs[4]]) { for(int k=0; k<5; k++) simplex[vs[5]][k]=ref[k]; scores[vs[5]]=r_score; }
            else { double con[5]; SVCJParams cp=*p; for(int k=0; k<5; k++) con[k]=c[k]+0.5*(simplex[vs[5]][k]-c[k]); check_constraints(&cp);
                double c_score=ukf_log_likelihood(returns, n_ret, &cp, NULL, NULL, theta_anchor);
                if(c_score>scores[vs[5]]) { for(int k=0; k<5; k++) simplex[vs[5]][k]=con[k]; scores[vs[5]]=c_score; }
                else { for(int i=1; i<6; i++) { int idx=vs[i]; SVCJParams sp=*p; for(int k=0; k<5; k++) simplex[idx][k]=simplex[vs[0]][k]+0.5*(simplex[idx][k]-simplex[vs[0]][k]); check_constraints(&sp); scores[idx]=ukf_log_likelihood(returns, n_ret, &sp, NULL, NULL, theta_anchor); } }
            }
        }
        int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
        if(scores[best] > global_best) { global_best = scores[best]; final_p.kappa=simplex[best][0]; final_p.theta=simplex[best][1]; final_p.sigma_v=simplex[best][2]; final_p.rho=simplex[best][3]; final_p.lambda_j=simplex[best][4]; }
    }
    *p = final_p;
    ukf_log_likelihood(returns, n_ret, p, out_spot_vol, out_jump_prob, theta_anchor);
    free(returns);
}

double normal_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
double bs_calc(double S, double K, double T, double r, double v, int type) {
    if(T < 1e-4) return (type==1)?fmax(S-K,0):fmax(K-S,0);
    double d1 = (log(S/K)+(r+0.5*v*v)*T)/(v*sqrt(T)); double d2 = d1 - v*sqrt(T);
    return (type==1) ? S*normal_cdf(d1)-K*exp(-r*T)*normal_cdf(d2) : K*exp(-r*T)*normal_cdf(-d2)-S*normal_cdf(-d1);
}
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    double lambda = p->lambda_j; double m = p->mu_j; double v_j = p->sigma_j; double lamp = lambda*(1.0+m);
    for(int i=0; i<n_opts; i++) {
        double val = 0.0;
        for(int k=0; k<12; k++) {
             double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
             double prob = exp(-lamp*expiries[i]) * pow(lamp*expiries[i], k) / fact;
             if(prob < 1e-8 && k>2) break;
             double rk = p->mu - lambda*m + (k*log(1.0+m))/expiries[i];
             double vk = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/expiries[i]);
             val += prob * bs_calc(s0, strikes[i], expiries[i], rk, vk, types[i]);
        }
        out_prices[i] = val;
    }
}