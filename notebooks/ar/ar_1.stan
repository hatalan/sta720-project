data {
  int<lower=1> T;
  vector[T] y;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  for (t in 2:T)
    y[t] ~ normal(alpha + beta * y[t-1], sigma);
}
