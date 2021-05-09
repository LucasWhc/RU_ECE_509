ALPHA = 0.01; 
BETA = 0.5;
MAXITERS = 30;
NTTOL = 1e-8;
n=5000;
x = zeros(n,1);
A=rand(n,n);
res = zeros(1,MAXITERS);
iter = 0;
for iter = 1:MAXITERS
    val = -sum(log(1-A*x)) - sum(log(1+x)) - sum(log(1-x)); 
    fprintf("\niter: %d,val: %d ",iter,val);
    res(1,iter) = val;
    d = 1./(1-A*x);
    grad = A'*d - 1./(1+x) + 1./(1-x);
    hess = A'*diag(d.^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);
    v = -hess\grad;
    fprime = grad'*v;

    t = 1; 
    while ( -sum(log(1-A*(x+t*v))) - sum(log(1-(x+t*v).^2)) > ...
    val + ALPHA*t*fprime )
        t = BETA*t;
    end
    x=x+t*v;
end
plot(res);                                                                                                                                      