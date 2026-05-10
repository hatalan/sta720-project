data {
  int<lower=1> T;
  int<lower=1> K;
  matrix[T, K] Y;
  
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=K> node1;
  array[N_edges] int<lower=1, upper=K> node2;
}

parameters {
  real gamma;               
  vector[K] c_zero;          
  real<lower=0> sigma_c;      

  real delta;              
  vector[K] phi_zero;         
  real<lower=0> sigma_phi;    
  
  vector<lower=0>[K] sigma;
}

transformed parameters {
  vector[K] c = gamma + sigma_c * c_zero;
  vector[K] beta = delta + sigma_phi * phi_zero;
}

model {
  gamma ~ normal(0, 1); 
  delta ~ normal(0.3, 0.3); 
  
  sigma_c ~ normal(0, 1); 
  sigma_phi ~ normal(0, 0.3); 
  
  c_zero ~ normal(0, 1);
  
  // ICAR prior
  target += -0.5 * dot_self(phi_zero[node1] - phi_zero[node2]);
  sum(phi_zero) ~ normal(0, 0.001 * K);
  
  sigma ~ normal(0, 1); 
  
  // Likelihood
  for (k in 1:K) {
    Y[2:T, k] ~ normal(c[k] + beta[k] * Y[1:(T-1), k], sigma[k]);
  }
}
