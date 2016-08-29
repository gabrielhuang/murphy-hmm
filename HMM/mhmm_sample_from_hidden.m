function [obs, hidden] = mhmm_sample_from_hidden(hidden, initial_prob, mu, Sigma, mixmat)
% See mhmm_sample

Q = length(initial_prob);
if nargin < 7, mixmat = ones(Q,1); end
O = size(mu,1);
T = size(hidden, 1);
numex = size(hidden, 2);
obs = zeros(O, T, numex);

for i=1:numex
  for t=1:T
    q = hidden(t,i);
    m = sample_discrete(mixmat(q,:), 1, 1);
    obs(:,t,i) =  gaussian_sample(mu(:,q,m), Sigma(:,:,q,m), 1);
  end
end
