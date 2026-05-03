data {
  int<lower=1> T;           // number of time periods
  int<lower=1> K;           // number of nodes
  matrix[T, K] Y;           // data
}

parameters {
  real gamma; // population intercept 
  real <lower=0, upper=1> delta; // population coefficient

  vector[K] c;              // node intercepts
  vector[K] beta;           // node coefficients
  vector<lower=0>[K] sigma; // node innovation standard deviation

  // Hyperparameters
  real<lower=0> lambda_c;       // tightness for intercepts
  real<lower=0> lambda_beta;    // tightness for coefficients
  real<lower=0> mu_sigma;       // mean of sigma hyperprior
  real<lower=0> tau_sigma;      // scale of sigma hyperprior
}

model {
  // Hyperpriors
  lambda_c ~ gamma(2, 0.75);
  lambda_beta ~ gamma(2, 0.75);
  mu_sigma ~ normal(0, 2);
  tau_sigma ~ normal(0, 1);

  // Hierarchical priors
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
