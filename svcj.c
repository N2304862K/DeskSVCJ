#include "svcj.h"
#include <float.h>

// --- Stat Helpers ---
double fast_erfc(double x) {
    double t = 1.0 / (1.0 + 0.5 * fabs(x));
    double tau = t * exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
}
double norm_cdf(double x) { return 0.5 * fast_erfc(-x * 0.70710678); }
double f_test_prob(double f, int df1, int df2) {
    if (f <= 0) return 1.0;
    double a = 2.0/(9.0*df1); double b = 2.0/(9.0*df2);
    double y = pow(f, 1.0/3.0);
    double z = ((1.0-b)*y - (1.0-a))/sqrt(b*y*y + a);
    return 2.0 * norm_cdf(-fabs(z));
}
double t_test_prob(double t, int df) { return 2.0 * norm_cdf(-fabs(t)); }

// --- Core ---
void compute_log_returns(double* ohlcv, int n_rows, double* out_returns) {
    for(int i=1; i<n_rows; i++) {
        double prev = ohlcv[(i-1)*N_COLS+3]; double curr = ohlcv[i*N_COLS+3];
        if(prev < 1e-9) prev = 1e-9;
        out_returns[i-1] = log(curr/prev);
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa<0.01) p->kappa=0.01; if(p->kappa>100.0) p->kappa=100.0;
    if(p->theta<1e-6) p->theta=1e-6; if(p->theta>50.0) p->theta=50.0;
    if(p->sigma_v<0.01) p->sigma_v=0.01; if(p->sigma_v>50.0) p->sigma_v=50.0;
    if(p->rho>0.999) p->rho=0.999; if(p->rho<-0.999) p->rho=-0.999;
    if(p->lambda_j<0.001) p->lambda_j=0.001; if(p->lambda_j>3000.0) p->lambda_j=3000.0;
    if(p->sigma_j<0.0001) p->sigma_j=0.0001;
    double feller = 2.0*p->kappa*p->theta;
    if(p->sigma_v*p->sigma_v > feller*20.0) p->sigma_v=sqrt(feller*20.0);
}

void estimate_initial_params(double* ohlcv, int n, double dt, SVCJParams* p) {
    double sum_gk=0;
    for(int i=0; i<n; i++) {
        double O=ohlcv[i*N_COLS]; double H=ohlcv[i*N_COLS+1];
        double L=ohlcv[i*N_COLS+2]; double C=ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        double hl=log(H/L); double co=log(C/O);
        double val=0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
        if(val>0) sum_gk+=val;
    }
    double rv=(sum_gk/n)/dt;
    p->mu=0; p->theta=rv; p->kappa=2.0; p->sigma_v=sqrt(rv);
    p->rho=-0.5; p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
    check_constraints(p);
}

double ukf_pure_likelihood(double* returns, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll=0; double v=p->theta; double var_j=p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-9) v_pred=1e-9;
        double y = returns[t] - (p->mu - 0.5*v_pred)*dt;
        
        double rob_var = (v_pred<1e-9)?1e-9:v_pred; rob_var*=dt;
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI))*exp(-0.5*y*y/rob_var);
        double tot_j = rob_var+var_j; 
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI))*exp(-0.5*yj*yj/tot_j);
        
        double prior = p->lambda_j*dt; if(prior>0.999) prior=0.999;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den<1e-25) den=1e-25;
        double post = (pdf_j*prior)/den;
        
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-9)v=1e-9; if(v>50.0)v=50.0;
        
        if(out_spot_vol) out_spot_vol[t]=sqrt(v_pred);
        if(out_jump_prob) out_jump_prob[t]=post;
        ll+=log(den);
    }
    return ll;
}

double obj_func(double* returns, int n, double dt, SVCJParams* p) {
    return ukf_pure_likelihood(returns, n, dt, p, NULL, NULL) - 0.05*(p->sigma_v*p->sigma_v);
}

void optimize_svcj(double* ohlcv, int n, double dt, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params(ohlcv, n, dt, p);
    double* ret = malloc((n-1)*sizeof(double));
    if(!ret) return;
    compute_log_returns(ohlcv, n, ret);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p;
        if(i==1) t.kappa*=1.3; if(i==2) t.theta*=1.3; if(i==3) t.sigma_v*=1.3;
        if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
        check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
        simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(ret, n-1, dt, &t);
    }
    for(int k=0; k<NM_ITER; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = obj_func(ret, n-1, dt, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; } 
        else {
             double con[5]; SVCJParams cp = *p; for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
             cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
             check_constraints(&cp);
             double c_score = obj_func(ret, n-1, dt, &cp);
             if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = con[d]; scores[vs[5]] = c_score; }
        }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2];
    p->rho=simplex[best][3];   p->lambda_j=simplex[best][4];
    
    if(out_spot_vol) ukf_pure_likelihood(ret, n-1, dt, p, out_spot_vol, out_jump_prob);
    free(ret);
}

// --- Fidelity Engine with Physics Export ---
void run_fidelity_scan(double* ohlcv, int total_len, double dt, FidelityMetrics* out) {
    int win_impulse = 30; 
    int win_gravity = win_impulse * 4;
    
    if (total_len < win_gravity) { out->is_valid=0; return; }
    
    SVCJParams p_grav;
    int start_grav = total_len - win_gravity;
    optimize_svcj(ohlcv + start_grav*N_COLS, win_gravity, dt, &p_grav, NULL, NULL);
    
    // EXPORT PHYSICS
    out->fit_theta = p_grav.theta;
    out->fit_kappa = p_grav.kappa;
    out->fit_sigma_v = p_grav.sigma_v;
    out->fit_rho = p_grav.rho;
    out->fit_lambda = p_grav.lambda_j;
    
    SVCJParams p_imp;
    double* imp_spot = malloc((win_impulse-1)*sizeof(double));
    int start_imp = total_len - win_impulse;
    optimize_svcj(ohlcv + start_imp*N_COLS, win_impulse, dt, &p_imp, imp_spot, NULL);
    
    double kinetic = imp_spot[win_impulse-2];
    out->energy_ratio = (kinetic * kinetic) / p_grav.theta;
    
    double* ret = malloc((win_impulse-1)*sizeof(double));
    compute_log_returns(ohlcv + start_imp*N_COLS, win_impulse, ret);
    double res_sum=0, res_sq=0;
    for(int i=0; i<win_impulse-1; i++) {
        double r = ret[i] - (p_grav.mu*dt);
        res_sum+=r; res_sq+=r*r;
    }
    double res_var = (res_sq - (res_sum*res_sum)/(win_impulse-1))/(win_impulse-2);
    double std_err = sqrt(res_var/(win_impulse-1));
    
    out->residue_bias = res_sum;
    out->f_p_value = f_test_prob(out->energy_ratio, win_impulse-1, win_gravity-1);
    double t_stat = (res_sum/(win_impulse-1)) / std_err;
    out->t_p_value = t_test_prob(t_stat, win_impulse-2);
    
    out->is_valid = (out->f_p_value < 0.05 && out->t_p_value < 0.10) ? 1 : 0;
    
    free(imp_spot); free(ret);
}

void run_instant_filter(double return_val, double dt, SVCJParams* p, double* state_var, InstantState* out) {
    double v_curr = *state_var;
    double v_pred = v_curr + p->kappa*(p->theta - v_curr)*dt;
    if(v_pred<1e-9) v_pred=1e-9;
    
    double y = return_val - (p->mu - 0.5*v_pred)*dt;
    
    double jump_var = p->lambda_j*(p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
    double total_std = sqrt((v_pred + jump_var)*dt);
    if(total_std<1e-9) total_std=1e-9;
    
    out->innovation_z_score = y / total_std;
    
    double S = v_pred*dt + (p->lambda_j*dt*jump_var);
    double K = (p->rho*p->sigma_v*dt)/S;
    double v_new = v_pred + K*y;
    if(v_new<1e-9) v_new=1e-9;
    
    out->current_spot_vol = sqrt(v_new);
    double pdf = (1.0/sqrt(2*M_PI*v_pred*dt))*exp(-0.5*y*y/(v_pred*dt));
    double pr = p->lambda_j*dt;
    out->current_jump_prob = pr / (pr + pdf*(1-pr));
    
    *state_var = v_new;
}