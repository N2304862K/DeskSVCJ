#include "svcj.h"
#include <float.h>

void compute_log_returns(double* ohlcv, int n, double* out) {
    for(int i=1; i<n; i++) {
        double p = ohlcv[(i-1)*N_COLS+3]; double c = ohlcv[i*N_COLS+3];
        if(p<1e-9) p=1e-9;
        out[i-1] = log(c/p);
    }
}

// Hard Constraints (Broad boundaries to prevent math errors, but allowing extreme moves)
void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa=0.1; if(p->kappa > 50.0) p->kappa=50.0;
    if(p->theta < 1e-6) p->theta=1e-6; if(p->theta > 20.0) p->theta=20.0; // Allow extreme vol
    if(p->sigma_v < 0.01) p->sigma_v=0.01; if(p->sigma_v > 10.0) p->sigma_v=10.0;
    if(p->rho > 0.99) p->rho=0.99; if(p->rho < -0.99) p->rho=-0.99;
    if(p->lambda_j < 0.001) p->lambda_j=0.001; if(p->lambda_j > 2000.0) p->lambda_j=2000.0;
    if(p->sigma_j < 0.0001) p->sigma_j=0.0001;
    
    // Minimal Feller check just to prevent NaN
    if (2.0*p->kappa*p->theta < p->sigma_v*p->sigma_v * 0.1) p->sigma_v *= 0.9;
}

// Pure UKF (No Priors, No Penalties)
double raw_log_likelihood(double* ret, int n, double dt, SVCJParams* p) {
    double ll=0; double v=p->theta;
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // Predict
        double v_pred = v + p->kappa*(p->theta - v)*dt;
        if(v_pred < 1e-8) v_pred = 1e-8;

        // Innovation
        double drift = (p->mu - 0.5*v_pred);
        double y = ret[t] - drift*dt;

        // Density Mix
        double rob_var = fmax(v_pred, 1e-6)*dt; // Absolute floor
        double pdf_d = (1.0/sqrt(rob_var*2*M_PI)) * exp(-0.5*y*y/rob_var);
        
        double tot_j = rob_var + var_j;
        double yj = y - p->mu_j;
        double pdf_j = (1.0/sqrt(tot_j*2*M_PI)) * exp(-0.5*yj*yj/tot_j);
        
        double prior = (p->lambda_j*dt > 0.99) ? 0.99 : p->lambda_j*dt;
        double den = pdf_j*prior + pdf_d*(1.0-prior);
        if(den < 1e-20) den = 1e-20;
        
        // Update
        double post = (pdf_j*prior)/den;
        double S = v_pred*dt + post*var_j;
        v = v_pred + (p->rho*p->sigma_v*dt/S)*y;
        if(v<1e-8)v=1e-8; if(v>10.0)v=10.0;
        
        ll += log(den);
    }
    return ll; // Pure Likelihood
}

// Garman-Klass for Seed
double get_gk_vol(double* ohlcv, int n, double dt) {
    double sum=0;
    for(int i=0; i<n; i++) {
        double O=ohlcv[i*N_COLS]; double H=ohlcv[i*N_COLS+1];
        double L=ohlcv[i*N_COLS+2]; double C=ohlcv[i*N_COLS+3];
        if(L<1e-9)L=1e-9; if(O<1e-9)O=1e-9;
        double hl=log(H/L); double co=log(C/O);
        sum += 0.5*hl*hl - (2.0*log(2.0)-1.0)*co*co;
    }
    return (sum/n)/dt;
}

// Multi-Start Optimization (Replacing Priors)
void optimize_snapshot_raw(double* ohlcv, int n, double dt, SVCJParams* p) {
    double* ret = malloc((n-1)*sizeof(double));
    compute_log_returns(ohlcv, n, ret);
    double seed_theta = get_gk_vol(ohlcv, n, dt);
    
    // Define Grid of Starting Points to find Global Minima
    SVCJParams candidates[MULTI_START_POINTS];
    
    // 1. Standard (GK Vol, Moderate Jumps)
    candidates[0].theta = seed_theta; candidates[0].lambda_j = 0.5; candidates[0].sigma_v = sqrt(seed_theta);
    // 2. High Vol / Low Jump
    candidates[1].theta = seed_theta*1.5; candidates[1].lambda_j = 0.1; candidates[1].sigma_v = sqrt(seed_theta)*2;
    // 3. Low Vol / High Jump (Crash Scenario)
    candidates[2].theta = seed_theta*0.5; candidates[2].lambda_j = 5.0; candidates[2].sigma_v = sqrt(seed_theta);
    // 4. Mean Reversion Heavy
    candidates[3].theta = seed_theta; candidates[3].lambda_j = 0.5; candidates[3].kappa = 10.0;
    // 5. Trend Heavy
    candidates[4].theta = seed_theta; candidates[4].lambda_j = 0.5; candidates[4].kappa = 0.5;

    // Common Defaults
    for(int i=0; i<MULTI_START_POINTS; i++) {
        candidates[i].mu = 0; candidates[i].rho = -0.6; 
        candidates[i].mu_j = 0; candidates[i].sigma_j = sqrt(seed_theta);
        if(candidates[i].kappa == 0) candidates[i].kappa = 3.0;
        check_constraints(&candidates[i]);
    }

    // Run Optimization for EACH candidate
    SVCJParams best_p;
    double best_ll = -1e15;

    for(int run=0; run<MULTI_START_POINTS; run++) {
        SVCJParams curr = candidates[run];
        
        // --- Nelder-Mead Loop (Simplified) ---
        int n_dim=5; double simplex[6][5]; double scores[6];
        // Init Simplex around candidate
        for(int i=0; i<=n_dim; i++) {
            SVCJParams t = curr;
            if(i==1) t.kappa*=1.2; if(i==2) t.theta*=1.2; if(i==3) t.sigma_v*=1.2;
            if(i==4) t.rho+=0.2;   if(i==5) t.lambda_j*=1.5;
            check_constraints(&t);
            simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v;
            simplex[i][3]=t.rho;   simplex[i][4]=t.lambda_j;
            scores[i] = raw_log_likelihood(ret, n-1, dt, &t);
        }
        
        // Iterations
        for(int k=0; k<NM_ITER; k++) {
            int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
            for(int i=0; i<6; i++) for(int j=i+1; j<6; j++) 
                if(scores[vs[j]] > scores[vs[i]]) { int x=vs[i]; vs[i]=vs[j]; vs[j]=x; }
                
            double c[5]={0};
            for(int i=0; i<5; i++) for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d];
            for(int d=0; d<5; d++) c[d]/=5.0;
            
            double ref[5]; SVCJParams rp = curr;
            for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
            rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
            check_constraints(&rp);
            double r_score = raw_log_likelihood(ret, n-1, dt, &rp);
            
            if(r_score > scores[vs[0]]) {
                 for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score;
            } else {
                 double con[5]; SVCJParams cp = curr;
                 for(int d=0; d<5; d++) con[d] = c[d] + 0.5*(simplex[vs[5]][d]-c[d]);
                 cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
                 check_constraints(&cp);
                 double c_score = raw_log_likelihood(ret, n-1, dt, &cp);
                 if(c_score > scores[vs[5]]) { for(int d=0; d<5; d++) simplex[vs[5]][d]=con[d]; scores[vs[5]]=c_score; }
            }
        }
        
        int b=0; for(int i=1; i<6; i++) if(scores[i]>scores[b]) b=i;
        curr.kappa=simplex[b][0]; curr.theta=simplex[b][1]; curr.sigma_v=simplex[b][2];
        curr.rho=simplex[b][3];   curr.lambda_j=simplex[b][4];
        curr.final_likelihood = scores[b];
        
        // Keep Global Best
        if(curr.final_likelihood > best_ll) {
            best_ll = curr.final_likelihood;
            best_p = curr;
        }
    }
    
    *p = best_p;
    free(ret);
}

// Statistical Test
double chi2_p(double x, int k) {
    if(x<=0) return 1.0;
    double s=2.0/9.0/k; double z=(pow(x/k,1.0/3.0)-(1.0-s))/sqrt(s);
    return 0.5*erfc(z*M_SQRT1_2);
}

void run_snapshot_test(double* ohlcv, int w_long, int w_short, double dt, SnapshotStats* out) {
    SVCJParams p_long, p_short;
    
    // 1. Fit Long (Global Truth)
    optimize_snapshot_raw(ohlcv, w_long, dt, &p_long);
    out->long_theta = p_long.theta;
    
    // 2. Fit Short (Local Truth)
    int offset = w_long - w_short;
    optimize_snapshot_raw(ohlcv + offset*N_COLS, w_short, dt, &p_short);
    out->short_theta = p_short.theta;
    
    // 3. Compare on Short Data
    double* ret_short = malloc((w_short-1)*sizeof(double));
    compute_log_returns(ohlcv + offset*N_COLS, w_short, ret_short);
    
    double L_constrained = raw_likelihood(ret_short, w_short-1, dt, &p_long);
    double L_unconstrained = p_short.final_likelihood; // Already calculated
    
    out->ll_ratio = 2.0 * (L_unconstrained - L_constrained);
    if(out->ll_ratio < 0) out->ll_ratio = 0;
    
    out->p_value = 1.0 - chi2_p(out->ll_ratio, 5);
    out->divergence = (p_short.theta - p_long.theta) / p_long.theta;
    
    free(ret_short);
}