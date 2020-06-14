load('okra-values.mat');


lambda1=700;
lambda2=700;

init=zeros(ht*width,1);% init = theta here
opts=struct('intensity',I_mat_low,'y_test',y_test,'sig',sig,'E_tmpl',E_tmpl,'mu_templ',mu_templ,'angles',angles,'W',W,'l',l,'q',q,'lambda',lambda2);
opts(1).intensity=I_mat_low;
opts(1).y_test=y_test;
opts(1).sig=sig;
opts(1).E_tmpl=E_tmpl;
opts(1).mu_templ=mu_templ;
opts(1).angles=angles;
opts(1).W=W;
opts(1).l=l;
opts(1).q=q;
opts(1).lambda=lambda2;

% doing only one optimisation here
result=fista_backtracking(@calc_f,@grad,init,opts,@calc_F);





function ftheta=calc_f(theta)
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    thet=radon(recons,angles);
    thet=exp(-thet);
    thet=I.*thet;
    num=y_test-thet;
    num=num.*num;
    den=thet+sig*sig;
    term1=num/den;
    term3=mu_templ+E_tmpl*E_tmpl'*(reshape(recons,[ht*width 1])-mu_templ);
    term3=reshape(term3, [ht width]);
    term3=recons-term3;
    term3=term3.*W;
    term3=norm(term3,'fro')^2;
    ftheta=term1+lam2*term3;
end
% function [ans_vec] = grad(I_mat,y_test,theta,sig,E_tmpl,mu_templ,angles,W,l,q)
function [ans_vec] = grad(theta,opts)
    I_mat=opts.intensity;
    y_test=opts.y_test;
    sig=opts.sig;
    E_tmpl=opts.E_tmpl;
    mu_templ=opts.mu_templ;
    angles=opts.angles;
    W=opts.W;
    l=opts.l;
    q=opts.q;
    % term1
    I_mat=reshape(I_mat,[l*q,1]);
    y=reshape(y_test,[l*q,1]);
    m=length(y);
    hw=length(theta);
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    P=radon(recons,angles);
    U1=zeros(m,m);
    for i=1:m
        term1=y(i)+I_mat(i)*exp(-1*P(i));
        term2=(y(i)-I_mat(i)*exp(-1*P(i)) -2*sig*sig);
        term3=I_mat(i);
        term5=(I_mat(i)*exp(-1*P(i)) + sig*sig);
        U1(i,i)=(term1*term2*term3)/term5;
    end
    matrix1=dct2(iradon(U1,angles));
    term1=zeros(hw,1);
    for i=1:m
        term1=term1+matrix1(:,i);
    end
    
    % term3
    alpha_templ=transpose(E_tmpl)*(recons-mu_templ); % the alpha for term 3
    [W1,~]=size(W);
    U2=zeros(W1,W1);
    left_term=W*recons;
    right_term=W*(mu_templ + E_tmpl*alpha_templ);
    for i=1:W1
        U2(i,i)=left_term(i) - right_term(i);
    end
    matrix2=dct2(transpose(W)*U2);
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