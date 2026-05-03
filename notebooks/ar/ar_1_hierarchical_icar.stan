data {
  int<lower=1> T;           // number of time periods
  int<lower=1> K;           // number of nodes
  matrix[T, K] Y;           // data

  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=K> node1; // node1[i] adjacent to node2[i]
  array[N_edges] int<lower=1, upper=K> node2; // and node1[i] < node2[i]
}

parameters {
  real gamma;                   // population intercept
  real<lower=0, upper=1> delta; // population AR coefficient

  vector[K] phi_c;             // ICAR spatial effects for intercepts
  vector[K] phi_beta;          // ICAR spatial effects for coefficients
  vector<lower=0>[K] sigma;    // node innovation standard deviation

  // Hyperparameters
  real<lower=0> tau_c;         // precision of ICAR for intercepts
  real<lower=0> tau_beta;      // precision of ICAR for coefficients
  real<lower=0> mu_sigma;      // mean of sigma hyperprior
  real<lower=0> tau_sigma;     // scale of sigma hyperprior
}

transformed parameters {
  vector[K] c = gamma + phi_c;
  vector[K] beta = delta + phi_beta;
}

model {
  // Population priors
  gamma ~ normal(0, 1);
  delta ~ beta(3.5, 2);

  // Hyperpriors
  tau_c ~ gamma(2, 0.75);
  tau_beta ~ gamma(2, 0.75);
  mu_sigma ~ normal(0, 2);
  tau_sigma ~ normal(0, 1);

  // ICAR priors on spatial deviations
  target += -0.5 * tau_c * dot_self(phi_c[node1] - phi_c[node2]);
  target += -0.5 * tau_beta * dot_self(phi_beta[node1] - phi_beta[node2]);

  // Sum-to-zero constraints for identifiability
  sum(phi_c) ~ normal(0, 0.001 * K);
  sum(phi_beta) ~ normal(0, 0.001 * K);

  // Sigma prior
  sigma ~ normal(mu_sigma, tau_sigma);

  // Likelihood
  for (k in 1:K) {
    for (t in 2:T) {
      Y[t, k] ~ normal(c[k] + beta[k] * Y[t-1, k], sigma[k]);
    }
  }
}
