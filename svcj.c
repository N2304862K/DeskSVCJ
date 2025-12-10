#include "svcj.h"
#include <float.h>

// --- Helpers (Stats, Sort) ---
int cmp(const void* a, const void* b) { double x=*(double*)a, y=*(double*)b; return (x>y)-(x<y); }
void sort_doubles(double* arr, int n) { qsort(arr, n, sizeof(double), cmp); }
void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.01) p->kappa = 0.01; if(p->kappa > 100.0) p->kappa = 100.0;
    if(p->theta < 1e-6) p->theta = 1e-6; if(p->theta > 100.0) p->theta = 100.0;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01; if(p->sigma_v > 50.0) p->sigma_v = 50.0;
    if(p->rho > 0.999) p->rho = 0.999; if(p->rho < -0.999) p->rho = -0.999;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001;
}

// --- Core Physics (SVCJ Optimizer) ---
double ukf_likelihood(double* ret, double* vol, int n, double dt, double avg_vol, SVCJParams* p) {
    double ll=0, v=p->theta; 
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    for(int t=0; t<n; t++) {
        double vol_scale = (avg_vol > 0) ? vol[t]/avg_vol : 1.0;
        if(vol_scale < 0.1) vol_scale = 0.1; if(vol_scale > 10.0) vol_scale = 10.0;
        double dt_eff = dt * vol_scale;
        double v_pred = v + p->kappa*(p->theta-v)*dt_eff;
        if(v_pred < 1e-9) v_pred = 1e-9;
        double y = ret[t] - (p->mu - 0.5*v_pred)*dt_eff;
        double tot_var = v_pred + (p->lambda_j * var_j);
        double step_std = sqrt(tot_var * dt_eff);
        if(step_std < 1e-9) step_std = 1e-9;
        double pdf = (1.0/(sqrt(2*M_PI)*step_std)) * exp(-0.5*y*y/(step_std*step_std));
        ll += log(pdf + 1e-25);
        double K = 0.1; v = v_pred + K * (y*y - tot_var*dt_eff);
        if(v < 1e-9) v = 1e-9;
    }
    return ll;
}

double obj_func(double* r, double* v, int n, double dt, double av, SVCJParams* p) {
    return ukf_likelihood(r, v, n, dt, av, p) - 0.05*p->sigma_v*p->sigma_v;
}

void optimize_single_window(double* ret, double* vol, int n, double dt, SVCJParams* p) {
    double sum_sq=0, sum_vol=0; for(int i=0; i<n; i++) { sum_sq+=ret[i]*ret[i]; sum_vol+=vol[i]; }
    double rv = (sum_sq/n)/dt; double av = sum_vol/n;
    p->mu=0; p->theta=rv; p->kappa=2.0; p->sigma_v=sqrt(rv); p->rho=-0.5; 
    p->lambda_j=0.5; p->mu_j=0; p->sigma_j=sqrt(rv);
    
    int n_dim=5; double simplex[6][5]; double scores[6];
    for(int i=0; i<=n_dim; i++) {
        SVCJParams t = *p; if(i==1) t.kappa*=1.3; if(i==2) t.theta*=1.3; check_constraints(&t);
        simplex[i][0]=t.kappa; simplex[i][1]=t.theta; simplex[i][2]=t.sigma_v; simplex[i][3]=t.rho; simplex[i][4]=t.lambda_j;
        scores[i] = obj_func(ret, vol, n, dt, av, &t);
    }
    for(int k=0; k<NM_ITER; k++) {
        int vs[6]; for(int j=0; j<6; j++) vs[j]=j;
        for(int i=0; i<6; i++) { for(int j=i+1; j<6; j++) { if(scores[vs[j]] > scores[vs[i]]) { int tmp=vs[i]; vs[i]=vs[j]; vs[j]=tmp; } } }
        double c[5]={0}; for(int i=0; i<5; i++) { for(int d=0; d<5; d++) c[d]+=simplex[vs[i]][d]; } for(int d=0; d<5; d++) c[d]/=5.0;
        double ref[5]; SVCJParams rp = *p; for(int d=0; d<5; d++) ref[d] = c[d] + 1.0*(c[d]-simplex[vs[5]][d]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4]; check_constraints(&rp);
        double r_score = obj_func(ret, vol, n, dt, av, &rp);
        if(r_score > scores[vs[0]]) { for(int d=0; d<5; d++) simplex[vs[5]][d] = ref[d]; scores[vs[5]] = r_score; }
    }
    int best=0; for(int i=1; i<6; i++) if(scores[i]>scores[best]) best=i;
    p->kappa=simplex[best][0]; p->theta=simplex[best][1]; p->sigma_v=simplex[best][2]; p->rho=simplex[best][3]; p->lambda_j=simplex[best][4];
}

void compute_log_returns(double* ohlcv, int n, double* out_ret, double* out_vol) {
    for(int i=1; i<n; i++) {
        double prev = ohlcv[(i-1)*N_COLS + 3];
        double curr = ohlcv[i*N_COLS + 3];
        out_vol[i-1] = ohlcv[i*N_COLS + 4];
        if(prev < 1e-9) prev = 1e-9;
        out_ret[i-1] = log(curr/prev);
    }
}

// --- INITIALIZATION ENGINE ---
EvolvingSystemState* initialize_system(double* ohlcv, int n, double dt, int n_particles) {
    EvolvingSystemState* state = (EvolvingSystemState*)malloc(sizeof(EvolvingSystemState));
    if (!state) return NULL;

    int windows[15]; double sigmas[15]; int count = 0;
    double curr = 30.0;
    while(curr <= n/2.0 && count < 15) { windows[count] = (int)curr; curr *= 1.4; count++; }
    
    double* ret = malloc(n*sizeof(double)); double* vol = malloc(n*sizeof(double));
    compute_log_returns(ohlcv, n, ret, vol);
    
    double min_sigma = 1e9; int natural_window = 0;
    for(int i=0; i<count; i++) {
        int w = windows[i]; int start = n - w;
        SVCJParams p;
        optimize_single_window(ret+start, vol+start, w-1, dt, &p);
        sigmas[i] = p.sigma_v;
        if(p.sigma_v < min_sigma) { min_sigma = p.sigma_v; natural_window = w; }
    }
    
    SVCJParams ensemble[5];
    for(int i=0; i<5; i++) {
        int w = natural_window + (i-2)*5; int start = n - w;
        optimize_single_window(ret+start, vol+start, w-1, dt, &ensemble[i]);
    }
    
    for(int j=0; j<8; j++) state->anchor.mean[j] = 0;
    for(int i=0; i<5; i++) {
        state->anchor.mean[0] += ensemble[i].kappa; state->anchor.mean[1] += ensemble[i].theta;
        state->anchor.mean[2] += ensemble[i].sigma_v; state->anchor.mean[3] += ensemble[i].rho;
        state->anchor.mean[4] += ensemble[i].lambda_j;
    }
    for(int j=0; j<5; j++) state->anchor.mean[j] /= 5.0;
    
    if (n_particles > MAX_PARTICLES) n_particles = MAX_PARTICLES;
    for(int i=0; i<n_particles; i++) {
        double z = sqrt(-2.0*log((double)rand()/RAND_MAX)) * cos(2.0*M_PI*((double)rand()/RAND_MAX));
        state->swarm[i].kappa = state->anchor.mean[0] + z*0.1;
        state->swarm[i].theta = state->anchor.mean[1] + z*0.01;
        state->swarm[i].sigma_v = state->anchor.mean[2] + z*0.05;
        state->swarm[i].rho = state->anchor.mean[3] + z*0.05;
        state->swarm[i].lambda_j = state->anchor.mean[4] + z*0.1;
        state->swarm[i].mu=0; state->swarm[i].mu_j=-0.05; state->swarm[i].sigma_j=0.05;
        state->swarm[i].weight = 1.0/n_particles;
    }
    
    state->n_particles = n_particles;
    state->dt = dt;
    double avg_vol=0; for(int i=0;i<n-1; i++) avg_vol+=vol[i];
    state->avg_volume = avg_vol/(n-1);
    
    free(ret); free(vol);
    return state;
}

void cleanup_system(EvolvingSystemState* state) {
    if (state) free(state);
}

// --- LIVE FILTER ENGINE ---
void run_system_step(EvolvingSystemState* state, double new_ret, double new_vol, InstantMetrics* out_metrics) {
    int n = state->n_particles; double dt = state->dt; double avg_vol = state->avg_volume;
    
    double total_weight = 0;
    for(int i=0; i<n; i++) {
        Particle p = state->swarm[i];
        double vol_scale = (avg_vol > 0) ? new_vol/avg_vol : 1.0;
        if(vol_scale < 0.1) vol_scale = 0.1; if(vol_scale > 10.0) vol_scale = 10.0;
        double dt_eff = dt * vol_scale;
        double exp_var = p.theta*dt_eff + p.lambda_j*(p.mu_j*p.mu_j + p.sigma_j*p.sigma_j);
        double diff = new_ret - (p.mu - 0.5*p.theta)*dt_eff;
        double likelihood = (1.0/sqrt(2*M_PI*exp_var)) * exp(-0.5*diff*diff/exp_var);
        state->swarm[i].weight = likelihood;
        total_weight += likelihood;
    }
    
    // Normalize & calc EV
    double ev_ret=0, ev_theta=0, entropy=0;
    for(int i=0; i<n; i++) {
        state->swarm[i].weight /= total_weight;
        double w = state->swarm[i].weight;
        ev_ret += w * state->swarm[i].mu;
        ev_theta += w * state->swarm[i].theta;
        if(w > 1e-9) entropy -= w * log(w);
    }
    out_metrics->expected_return = ev_ret;
    out_metrics->expected_vol = sqrt(ev_theta);
    out_metrics->swarm_entropy = entropy;
    
    // Resample
    Particle* new_swarm = malloc(n * sizeof(Particle));
    int idx = rand() % n; double beta = 0.0, max_w = 0;
    for(int i=0; i<n; i++) if(state->swarm[i].weight > max_w) max_w = state->swarm[i].weight;
    for(int i=0; i<n; i++) {
        beta += (double)rand()/RAND_MAX * 2.0 * max_w;
        while(beta > state->swarm[idx].weight) {
            beta -= state->swarm[idx].weight;
            idx = (idx+1)%n;
        }
        new_swarm[i] = state->swarm[idx];
    }
    
    // Mutate & Gravitate
    for(int i=0; i<n; i++) {
        new_swarm[i].theta = 0.9*new_swarm[i].theta + 0.1*state->anchor.mean[1];
        double z = sqrt(-2.0*log((double)rand()/RAND_MAX)) * cos(2.0*M_PI*((double)rand()/RAND_MAX));
        new_swarm[i].kappa *= (1.0 + 0.01*z);
        check_constraints((SVCJParams*)&new_swarm[i]);
        state->swarm[i] = new_swarm[i];
    }
    free(new_swarm);
    
    // Escape Velocity (Mahalanobis Distance)
    double diff_theta = ev_theta - state->anchor.mean[1];
    out_metrics->escape_velocity = fabs(diff_theta) / 0.01; // Assuming 0.01 std err on theta
}