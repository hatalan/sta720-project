data {
  int<lower=1> T;           // number of time periods
  int<lower=1> K;           // number of nodes
  matrix[T, K] Y;           // data

  int<lower=0> N;
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=N> node1; // node1[i] adjacent to node2[i]
  array[N_edges] int<lower=1, upper=N> node2; // and node1[i] < node2[i]
}

parameters {
  vector[K] c;              // node intercepts
  vector[K] beta;           // node coefficients
  vector<lower=0>[K] sigma; // node innovation standard deviation

  // Hyperparameters
  real<lower=0> lambda_c;       // tightness for intercepts
  real<lower=0> lambda_beta;    // tightness for coefficients
  real<lower=0> mu_sigma;       // mean of sigma hyperprior
  real<lower=0> tau_sigma;      // scale of sigma hyperprior

  // ICAR parameter
  vector[N] phi;
}

model {
  // Hyperpriors
  lambda_c ~ gamma(2, 0.75);
  lambda_beta ~ gamma(2, 0.75);
  mu_sigma ~ normal(0, 2);
  tau_sigma ~ normal(0, 1);

  // ICAR prior
  target += -0.5 * dot_self(phi[node1] - phi[node2]);
  sum(phi) ~ normal(0, 0.01 * N);

  // Minnesota-like priors
  c ~ normal(0, lambda_c);              // no constant shock
  beta ~ normal(1, lambda_beta);        // shrink towards 1 (random-walk)     
  sigma ~ normal(mu_sigma, tau_sigma);

  // Likelihood
  for (k in 1:K) {
    for (t in 2:T) {
      Y[t, k] ~ normal(c[k] + beta[k] * Y[t-1, k], sigma[k]);
    }
  }
}