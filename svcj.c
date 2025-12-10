#include "svcj.h"
#include <float.h>

// --- MATH UTILS ---
int cmp(const void* a, const void* b) {
    double x=*(double*)a, y=*(double*)b;
    return (x<y)?-1:(x>y);
}
double fast_erfc(double x) {
    double t = 1.0 / (1.0 + 0.5 * fabs(x));
    double tau = t * exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
}

double calc_hurst(double* data, int n) {
    if(n<20) return 0.5;
    double m=0; for(int i=0;i<n;i++) m+=data[i]; m/=n;
    double ss=0; for(int i=0;i<n;i++) ss+=(data[i]-m)*(data[i]-m);
    double std=sqrt(ss/n); if(std<1e-9) return 0.5;
    double maxd=-1e9, mind=1e9, c=0;
    for(int i=0;i<n;i++) { c+=(data[i]-m); if(c>maxd)maxd=c; if(c<mind)mind=c; }
    return log((maxd-mind)/std)/log((double)n);
}

// --- CORE ---
void compute_log_returns(double* ohlcv, int n, double* out) {
    for(int i=1; i<n; i++) {
        double p0 = ohlcv[(i-1)*N_COLS + 3];
        double p1 = ohlcv[i*N_COLS + 3];
        out[i-1] = log(p1/p0);
    }
}

// --- OPTIMIZATION (Condensed for brevity) ---
double ukf_likelihood(double* ret, int n, double dt, SVCJParams* p) {
    double ll=0; double v=p->theta; 
    for(int t=0; t<n; t++) {
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred<1e-9) v_pred=1e-9;
        double y = ret[t] - (p->mu - 0.5*v_pred)*dt;
        double s = v_pred*dt + p->lambda_j*dt*(p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double pdf = (1.0/sqrt(2*M_PI*s))*exp(-0.5*y*y/s);
        ll += log(pdf + 1e-20);
        v = v_pred + (p->rho*p->sigma_v*dt/s)*y;
        if(v<1e-9)v=1e-9; if(v>20.0)v=20.0;
    }
    return ll;
}

void optimize_svcj(double* ret, int n, double dt, SVCJParams* p) {
    double sum_sq=0; for(int i=0;i<n;i++) sum_sq+=ret[i]*ret[i];
    p->theta = (sum_sq/n)/dt; 
    p->mu=0; p->kappa=3.0; p->sigma_v=0.5; p->rho=-0.5; 
    p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(p->theta);
    
    int nd=5; double sim[6][5]; double sc[6];
    for(int i=0;i<=nd;i++) {
        SVCJParams t = *p;
        if(i==1)t.kappa*=1.2; if(i==2)t.theta*=1.2; if(i==3)t.sigma_v*=1.2;
        if(i==4)t.rho+=0.2; if(i==5)t.lambda_j*=1.2;
        sim[i][0]=t.kappa; sim[i][1]=t.theta; sim[i][2]=t.sigma_v; 
        sim[i][3]=t.rho; sim[i][4]=t.lambda_j;
        sc[i] = ukf_likelihood(ret, n, dt, &t);
    }
    
    for(int k=0;k<NM_ITER;k++) {
        int vs[6]; for(int j=0;j<6;j++)vs[j]=j;
        for(int i=0;i<6;i++) for(int j=i+1;j<6;j++) if(sc[vs[j]]>sc[vs[i]]) {int tp=vs[i]; vs[i]=vs[j]; vs[j]=tp;}
        double c[5]={0}; for(int i=0;i<5;i++) for(int d=0;d<5;d++) c[d]+=sim[vs[i]][d]; for(int d=0;d<5;d++) c[d]/=5;
        SVCJParams rp=*p; double ref[5];
        for(int d=0;d<5;d++) ref[d]=c[d]+(c[d]-sim[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        double rsc=ukf_likelihood(ret, n, dt, &rp);
        if(rsc>sc[vs[0]]) { for(int d=0;d<5;d++) sim[vs[5]][d]=ref[d]; sc[vs[5]]=rsc; }
        else {
            SVCJParams cp=*p; 
            for(int d=0;d<5;d++) sim[vs[5]][d] = c[d] + 0.5*(sim[vs[5]][d]-c[d]);
            cp.kappa=sim[vs[5]][0]; cp.theta=sim[vs[5]][1]; cp.sigma_v=sim[vs[5]][2]; 
            cp.rho=sim[vs[5]][3]; cp.lambda_j=sim[vs[5]][4]; 
            sc[vs[5]] = ukf_likelihood(ret, n, dt, &cp);
        }
    }
    int b=0; for(int i=1;i<6;i++) if(sc[i]>sc[b]) b=i;
    p->kappa=sim[b][0]; p->theta=sim[b][1]; p->sigma_v=sim[b][2]; p->rho=sim[b][3]; p->lambda_j=sim[b][4];
}

// --- PHYSICS ENGINES ---
void fit_gravity_physics(double* ohlcv, int n, double dt, SVCJParams* out) {
    // A. Trend (Drift)
    double sum_x=0, sum_y=0, sum_xy=0, sum_x2=0;
    for(int i=0; i<n; i++) {
        double ln_p = log(ohlcv[i*N_COLS + 3]);
        double t = i * dt;
        sum_x += t; sum_y += ln_p;
        sum_xy += t*ln_p; sum_x2 += t*t;
    }
    out->mu = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x*sum_x);
    
    // B. Variance (Detrended)
    double* detrended_ret = malloc((n-1)*sizeof(double));
    for(int i=1; i<n; i++) {
        double raw_ret = log(ohlcv[i*N_COLS+3] / ohlcv[(i-1)*N_COLS+3]);
        detrended_ret[i-1] = raw_ret - (out->mu * dt);
    }
    
    optimize_svcj(detrended_ret, n-1, dt, out);
    free(detrended_ret);
}

void test_causal_cone(double* impulse_prices, int n, double dt, SVCJParams* grav, CausalStats* out) {
    double max_z = 0.0;
    double p0 = impulse_prices[0];
    
    // 1. Causal Cone Test
    for(int i=1; i<n; i++) {
        double t = i * dt;
        double ln_p_exp = log(p0) + (grav->mu * t);
        double structural_std = sqrt(grav->theta * t);
        if (structural_std < 1e-9) structural_std = 1e-9;
        
        double ln_p_act = log(impulse_prices[i]);
        double z = (ln_p_act - ln_p_exp) / structural_std;
        
        if (fabs(z) > fabs(max_z)) {
            max_z = z;
        }
    }
    
    out->max_deviation = max_z;
    out->p_value = fast_erfc(fabs(max_z) * 0.70710678);
    out->is_breakout = (fabs(max_z) > 2.24) ? 1 : 0;
    
    // 2. Confirmation Factors (The Physics Payload)
    double* ret_imp = malloc((n-1)*sizeof(double));
    for(int i=1; i<n; i++) ret_imp[i-1] = log(impulse_prices[i]/impulse_prices[i-1]);
    
    // Hurst
    out->hurst_exponent = calc_hurst(ret_imp, n-1);
    
    // Residue Bias
    double bias = 0;
    for(int i=0; i<n-1; i++) bias += (ret_imp[i] - (grav->mu * dt));
    out->residue_bias = bias;

    // Energy Ratio
    double sum_sq = 0;
    for(int i=0; i<n-1; i++) sum_sq += ret_imp[i]*ret_imp[i];
    double realized_var = (sum_sq/(n-1))/dt;
    out->energy_ratio = realized_var / grav->theta;
    
    free(ret_imp);
}