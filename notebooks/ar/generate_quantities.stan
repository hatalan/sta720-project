data {
  int<lower=1> T;
  int<lower=1> K;
  matrix[T, K] Y;

  int<lower=1> P;            // number of seasonal predictors
  matrix[T, P] X;            // seasonal design matrix
  
  int<lower=1> N_check;
  array[N_check] int<lower=1, upper=K> check_idx;
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

generated quantities {
  matrix[T, N_check] y_rep;
  
  for (i in 1:N_check) {
    int k = check_idx[i];
    y_rep[1, i] = Y[1, k];
    for (t in 2:T) {
      real mu = c[k] + beta[k] * y_rep[t-1, i] + X[t] * seas;
      y_rep[t, i] = normal_rng(mu, sigma[k]);
    }
  }
}