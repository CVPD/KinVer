function sim = cos_sim(Xa, Xb)   
%% cosine similarity

N = size(Xa, 2);
sim = zeros(N, 1);
for ii = 1:N
    x_a = Xa(:, ii);
    x_b = Xb(:, ii);
    sim(ii) = (x_a' * x_b) / (norm(x_a) * norm(x_b));
end
sim = (sim + 1) / 2; % sacle in range [0 1]