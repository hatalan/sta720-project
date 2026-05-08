data {
  int<lower=1> T;
  int<lower=1> K;
  matrix[T, K] Y;

  int<lower=1> P;            // number of seasonal predictors
  matrix[T, P] X;            // seasonal design matrix
  
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=K> node1;
  array[N_edges] int<lower=1, upper=K> node2;
}

parameters {
  real gamma;                   // population mean intercept
  vector[K] c_zero;              // station intercept deviations
  real<lower=0> sigma_c;        // SD on intercept deviations

  real delta;                   // population mean AR coefficient
  vector[K] phi_zero;            // station spatial coefficient deviations
  real<lower=0> sigma_phi;      // SD on spatial coefficient deviations
  
  vector<lower=0>[K] sigma;

  vector[P] seas;
}

transformed parameters {
  vector[K] c = gamma + sigma_c * c_zero;
  vector[K] beta = delta + sigma_phi * phi_zero;
}

model {
  gamma ~ normal(0, 1); // population mean intercept hyperprior
  delta ~ normal(0.3, 0.3); // population mean AR coefficient hyperprior
  
  sigma_c ~ normal(0, 1); // population intercept SD hyperprior
  sigma_phi ~ normal(0, 0.3); // population AR coefficient SD hyperprior
  
  c_zero ~ normal(0, 1);
  
  // ICAR prior
  target += -0.5 * dot_self(phi_zero[node1] - phi_zero[node2]);
  sum(phi_zero) ~ normal(0, 0.001 * K);
  
  sigma ~ normal(0, 1); // simple half-normal prior on innovation

  seas ~ normal(0, 0.5);
  
  // Likelihood
  for (k in 1:K) {
    Y[2:T, k] ~ normal(c[k] + beta[k] * Y[1:(T-1), k] + X[2:T] * seas, sigma[k]);
  }
}