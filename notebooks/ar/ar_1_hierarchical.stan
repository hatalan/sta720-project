data {
  int<lower=1> T;          
  int<lower=1> K;           
  matrix[T, K] Y;        
}

parameters {
  real gamma; 
  real <lower=0, upper=1> delta; 

  vector[K] c;           
  vector[K] beta;         
  vector<lower=0>[K] sigma; 

  real<lower=0> lambda_c;      
  real<lower=0> lambda_beta;    
  real<lower=0> mu_sigma;    
  real<lower=0> tau_sigma;     
}

model {
  lambda_c ~ gamma(2, 0.75);
  lambda_beta ~ gamma(2, 0.75);
  mu_sigma ~ normal(0, 2);
  tau_sigma ~ normal(0, 1);

  gamma ~ normal(0, 1);
  delta ~ beta(3.5, 2);
  c ~ normal(gamma, lambda_c);             
  beta ~ normal(delta, lambda_beta);            
  sigma ~ normal(mu_sigma, tau_sigma);

  // Likelihood
  for (k in 1:K) {
    for (t in 2:T) {
      Y[t, k] ~ normal(c[k] + beta[k] * Y[t-1, k], sigma[k]);
    }
  }
}
