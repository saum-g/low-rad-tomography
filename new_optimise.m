load('okra-values.mat');

lambda1=700;
lambda2=700;
I_low=10;
I_mat_low=ones(l,q)*I_low;
sig=1;
W=0;

init=zeros(ht*width,1);% init = theta here
global opts
opts=struct('ht',ht,'width',width,'intensity',I_mat_low,'y_test',y_test,'sig',sig,'E_tmpl',E_tmpl,'mu_templ',mu_templ,'angles',angles,'W',W,'l',l,'q',q,'lambda',lambda1,'lambda2',lambda2,'alpha_tmpl',E_tmpl'*(-mu_templ));
opts(1).ht=ht;
opts(1).width=width;
opts(1).intensity=I_mat_low;
opts(1).y_test=y_test;
opts(1).sig=sig;
opts(1).E_tmpl=E_tmpl;
opts(1).mu_templ=mu_templ;
opts(1).angles=angles;
opts(1).W=W;
opts(1).l=l;
opts(1).q=q;
opts(1).lambda1=lambda1;
opts(1).lambda2=lambda2;
opts(1).alpha_tmpl=E_tmpl'*(-mu_templ);

% doing only one optimisation here
% result=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);

% alternating minimisation
old_value=calc_f(init);
recons_theta=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
new_value=calc_f(recons_theta);
while new_value-old_value>1
    opts.alpha_tmpl=fist_backtracking(@calc_term3,@grad_t3,opts.alpha_tmpl,opts,@calc_term3);
    old_value=new_value;
    recons_theta=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
    new_value=calc_f(recons_theta);
end


reconstruction=idct2(reshape(recons_theta,[ht,width]));

% determining lambda1
lambda1=0.01;
opts.lambda1=lambda1;
theta_initial=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
I_mat_low=reshape(I_mat_low,[l*q,1]);
recons=reshape(theta_initial,[ht,width]);
recons=idct2(recons);
term_exp=radon(recons,angles);
R=(y_test-I.*exp(-term_exp))/sqrt(I.*exp(-term_exp)+sig*sig);
m=length(R);
R_val=norm(R,2);
D=abs(R-sqrt(m));
D_prev=R_val;
error_limit=0.16;
while(D>error_limit && D<D_prev)
    D_prev=D;
    lambda1=lambda1+0.01;
    opts.lambda1=lambda1;
    theta_initial=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
    I_mat_low=reshape(I_mat_low,[l*q,1]);   
    recons=reshape(theta_initial,[ht,width]);
    recons=idct2(recons);
    term_exp=radon(recons,angles);
    R=(y_test-I.*exp(-term_exp))/sqrt(I.*exp(-term_exp)+sig*sig);
    m=length(R);
    R_val=norm(R,2);
    D=abs(R-sqrt(m));
end
if(D>=D_prev)
    lambda1=lambda1-0.01;
    opts.lambda1=lambda1;
end

function ftheta=calc_f(theta)
    global opts
    ht=opts.ht;
    width=opts.width;
    angles=opts.angles;
    I_mat=opts.intensity;
    y_test=opts.y_test;
    sig=opts.sig;
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    W=opts.W;
    lambda2=opts.lambda2;
    alpha_tmpl=opts.alpha_tmpl;
    
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    thet=radon(recons,angles);
    thet=exp(-thet);
    thet=I_mat.*thet;
    num=y_test-thet;
    num=num.*num;
    den=thet+sig*sig;
    term1=num/den;
    term3=mu_templ+E_tmpl*alpha_tmpl;
    term3=reshape(term3, [ht width]);
    term3=recons-term3;
    term3=term3.*W;
    term3=norm(term3,'fro')^2;
    ftheta=term1+lambda2*term3;
end
% function [ans_vec] = grad(I_mat,y_test,theta,sig,E_tmpl,mu_templ,angles,W,l,q)
function [ans_vec] = grad(theta)
    global opts
    I_mat=opts.intensity;
    y_test=opts.y_test;
    sig=opts.sig;
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    angles=opts.angles;
    W=opts.W;
    l=opts.l;
    q=opts.q;
    ht=opts.ht;
    width=opts.width;
    lambda2=opts.lambda2;
    alpha_tmpl=opts.alpha_tmpl;
    
    % term1
    I_mat=reshape(I_mat,[l*q,1]);
    y=reshape(y_test,[l*q,1]);
    m=length(y);
    hw=length(theta);
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    P=radon(recons,angles);
    U1=zeros(m,1);
    for i=1:m
        term1=y(i)+I_mat(i)*exp(-1*P(i));
        term2=(y(i)-I_mat(i)*exp(-1*P(i)) -2*sig*sig);
        term3=I_mat(i);
        term5=(I_mat(i)*exp(-1*P(i)) + sig*sig);
        U1(i)=(term1*term2*term3)/term5;
    end
    V1=reshape(U1,[l,q]);
    matrix1=dct2(iradon(V1,angles));
    disp(size(matrix1))
    term1=zeros(hw,1);
    for i=1:m
        term1=term1+matrix1(:,i);
    end

    % term3
    [W1,~]=size(W);
    U2=zeros(W1,1);
    left_term=W*recons;
    right_term=W*(mu_templ + E_tmpl*alpha_tmpl);
    for i=1:W1
        U2(i,i)=left_term(i) - right_term(i);
    end
    V2=reshape(U2,[W1,~]);
    matrix2=dct2(transpose(W)*V2);
    term2=zeros(hw,1);
    for i=1:W1
        term2=term2+matrix2(:,i);
    end
    
    % final value
    ans_vec=term1+2*lambda2*term2;
end
function Ftheta=calc_F(theta)
    Ftheta=calc_f(theta);
    Ftheta=Ftheta+lambda1*norm(theta,1);
end