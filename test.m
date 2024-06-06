data = load("data.mat");
M = data.M

[V,D] = joint_diag(M,1e-8);
V
D