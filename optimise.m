load('okra-values.mat');

lambda1=0.1;
lambda2=700;
% I_low=10;
% I_mat_low=ones(l,q)*I_low;
% sig=1;
% W=zeros(135,135);
% y_test=zeros(l,q);
% init=zeros(ht*width,1);% init = theta here
init=rand(ht,width);
temp=idct2(init);
temp(temp<0)=0;
init=dct2(temp);
init=reshape(init,[ht*width 1]);
init=(init/(sum(init,'all')))*normalised_sum;
% init=reshape(dct2(x_test(:,:,30)),[ht*width 1]);
global opts
% opts=struct('ht',ht,'width',width,'intensity',I_mat_low,'y_test',y_test,'sig',sig,'E_tmpl',E_tmpl,'mu_templ',mu_templ,'angles',angles,'W',W,'l',l,'q',q,'lambda',lambda1,'lambda2',lambda2,'alpha_tmpl',E_tmpl'*(-mu_templ),'verbose',true,'theta_recons',init,'tol',1e-6);
opts=struct('ht',ht,'width',width,'intensity',I_irr,'y_test',y_test_irr,'sig',sig,'E_tmpl',E_tmpl,'mu_templ',mu_templ,'angles',angles,'W',W,'l',l,'q',q,'lambda',lambda1,'lambda2',lambda2,'alpha_tmpl',E_tmpl'*(-mu_templ),'verbose',true,'theta_recons',init,'tol',1e-6);

opts(1).ht=ht;
opts(1).width=width;
% opts(1).intensity=I_mat_low;
% opts(1).y_test=y_test;
opts(1).intensity=I_irr;
opts(1).y_test=y_test_irr;
opts(1).sig=sig;
opts(1).E_tmpl=E_tmpl;
opts(1).mu_templ=mu_templ;
opts(1).angles=angles;
opts(1).W=W;
opts(1).l=l;
opts(1).q=q;
opts(1).lambda=lambda1;
opts(1).lambda2=lambda2;
opts(1).alpha_tmpl=E_tmpl'*(init-mu_templ);
opts(1).verbose=true;
opts(1).theta_recons=init;
opts(1).tol=5e-6;
% calc_f(init);
% my_check_grad(@calc_f,@grad,init);

% doing only one optimisation here
% result=fista_backtracking(@calc_f,@grad,opts.theta_recons,opts,@calc_F);
% rec_img=idct2(result);
% rec_img(rec_img<0)=0;
% result=dct2(rec_img);
% choose between the one below and the one above.

% alternating minimisation
old_value=calc_f(opts.theta_recons);
rec_theta=my_fista_backtracking(@calc_f,@grad,opts.theta_recons,opts,@calc_F);
rec_theta=reshape(rec_theta, [ht width]);
rec_img=idct2(rec_theta);
rec_img(rec_img<0)=0;
rec_theta=dct2(rec_img);
rec_theta=reshape(rec_theta, [ht*width 1]);
opts.theta_recons=rec_theta;
opts.lambda=0;
opts.alpha_tmpl=fista_backtracking(@calc_term3,@gradt3,opts.alpha_tmpl,opts,@calc_term3);
opts.lambda=lambda1;
new_value=calc_f(opts.theta_recons);
while old_value-new_value>1e-4
    rec_theta=my_fista_backtracking(@calc_f,@grad,opts.theta_recons,opts,@calc_F);
    rec_theta=reshape(rec_theta, [ht width]);
    rec_img=idct2(rec_theta);
    rec_img(rec_img<0)=0;
    rec_theta=dct2(rec_img);
    rec_theta=reshape(rec_theta, [ht*width 1]);
    opts.theta_recons=rec_theta;
    opts.lambda=0;
    opts.alpha_tmpl=fista_backtracking(@calc_term3,@gradt3,opts.alpha_tmpl,opts,@calc_term3);
    opts.lambda=lambda1;
    old_value=new_value;
    new_value=calc_f(opts.theta_recons);
end


% reconstruction=idct2(reshape(recons_theta,[ht,width]));

W_temp=W;

% determining lambda1
lambda1=0.01;
opts.lambda1=lambda1;
W=zeros(ht,width);
opts.W=W;
theta_initial=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
recons=reshape(theta_initial,[ht,width]);
recons=idct2(recons);
term_exp=radon(recons,angles);
R=(y_test-I_mat_low.*exp(-term_exp))/sqrt(I_mat_low.*exp(-term_exp)+sig*sig);
R=reshape(R,[l*q,1]);
m=length(R);
R_val=norm(R,2);
D=abs(R_val-sqrt(m));
D_prev=R_val;
error_limit=0.16;
while(D>error_limit && D<D_prev)
    disp(lambda1,D)
    D_prev=D;
    lambda1=lambda1+0.01;
    opts.lambda1=lambda1;
    theta_initial=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);
    recons=reshape(theta_initial,[ht,width]);
    recons=idct2(recons);
    term_exp=radon(recons,angles);
    R=(y_test-I_mat_low.*exp(-term_exp))/sqrt(I_mat_low.*exp(-term_exp)+sig*sig);
    R=reshape(R,[l*q,1]);
    R_val=norm(R,2);
    D=abs(R_val-sqrt(m));
end
if(D>D_prev)
    lambda1=lambda1-0.01;
    opts.lambda1=lambda1;
end
W=W_temp;
opts.W=W;



function ftheta=calc_f(theta)
    global opts
    ht=opts.ht;
    width=opts.width;
    angles=opts.angles;
    I_mat_low=opts.intensity;
    y_test=opts.y_test;
    sig=opts.sig;
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    W=opts.W;
    lambda2=opts.lambda2;
    alpha_tmpl=opts.alpha_tmpl;

    theta=reshape(theta,[ht,width]);
    temp=idct2(theta);
    temp(temp<0)=0;
    theta=dct2(temp);
    recons=theta;
    recons=idct2(recons);
%     disp(size(recons)) - 135,135
    thet=radon(recons,angles);
%     disp(size(thet)) - 193,360
    thet=exp(-1*thet);
    thet=I_mat_low.*thet;
    num=y_test.*I_mat_low-thet;
%     num=y_test-thet;
    num=num.*num;
    den=thet+sig*sig;
    term1=num./den;
    term1=sum(term1,'all');
    term3=mu_templ+E_tmpl*alpha_tmpl;
    term3=reshape(term3, [ht, width]);
    term3=recons-term3;
    term3=term3.*W;
    term3=norm(term3,'fro')^2;
    ftheta=term1+lambda2*term3;
end
% function [ans_vec] = grad(I_mat_low,y_test,theta,sig,E_tmpl,mu_templ,angles,W,l,q)
function [ans_vec] = grad(theta)
    global opts
    I_mat_low=opts.intensity;
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
    
    temp=reshape(theta,[ht width]);
    temp=idct2(temp);
    temp(temp<0)=0;
    theta=dct2(temp);
    theta=reshape(theta,[ht*width 1]);
    % term1
    I_mat_low=reshape(I_mat_low,[l*q,1]);
    y=reshape(y_test,[l*q,1]);
    y=y.*I_mat_low;
    m=length(y);
    hw=length(theta);
    recons=reshape(theta,[ht,width]);
%     disp(size(recons))
    recons=idct2(recons);
%     disp(size(recons))
    P=radon(recons,angles);
%    disp(m)
%     disp(size(P))
%     disp(size(iradon(P,angles)))
    P=reshape(P,[m,1]);
%     disp(size(P))
    term3=I_mat_low.*exp(-P);
    term1=y-term3;
    term5=term3+sig*sig;
    term5=term5.*term5;
    term2=y+term3+2*sig*sig;
    U1=term1.*term3.*term2./term5;
%     for i=1:m
%         term1=y(i)-I_mat_low(i)*exp(-1*P(i));
%         term2=(y(i)+I_mat_low(i)*exp(-1*P(i)) +2*sig*sig);
%         term3=I_mat_low(i)*exp(-1*P(i));
%         term5=(I_mat_low(i)*exp(-1*P(i)) + sig*sig);
%         U1(i)=(term1*term2*term3)/term5;
%     end
    V1=reshape(U1,[l,q]);
    matrix1=iradon(V1,angles,'None',ht); % error here
%     disp(size(matrix1))
%     [f,g]=size(matrix1);
%     matrix1=matrix1(2:f,2:g);
    matrix1=dct2(matrix1);
%     disp(size(matrix1))
    t1=reshape(matrix1,[hw,1]);
    
    % term3
    [ht,width]=size(W);
    left_term=W.*recons;
    right_term=W.*reshape((mu_templ + E_tmpl*alpha_tmpl),[ht,width]);
    U2=W.*(left_term-right_term);
    matrix2=dct2(U2);
    t2=reshape(matrix2,[hw,1]);
    
    % final value
    ans_vec=t1+2*lambda2*t2;
end

function Ftheta=calc_F(theta)
    global opts
    lambda=opts.lambda;
    Ftheta=calc_f(theta);
    Ftheta=Ftheta+lambda*norm(theta,1);
end

function t3=calc_term3(alpha)
    global opts
    ht=opts.ht;
    width=opts.width;
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    W=opts.W;
    theta=opts.theta_recons;
    
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    term3=mu_templ+E_tmpl*alpha;
    term3=reshape(term3, [ht, width]);
    term3=recons-term3;
    term3=term3.*W;
    t3=norm(term3,'fro')^2;
end

function [ans_vec]=gradt3(alpha)
    global opts
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    W=opts.W;
    ht=opts.ht;
    width=opts.width;
    theta=opts.theta_recons;
    
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    [ht,width]=size(W);
    left_term=W.*recons;
    right_term=W.*reshape((mu_templ + E_tmpl*alpha),[ht,width]);
    U2=W.*(left_term-right_term);
    U2=reshape(U2,[ht*width,1]);
    ans_vec=E_tmpl'*U2;
end