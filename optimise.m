load('okra-values.mat');


lambda1=700;
lambda2=700;

init=zeros(ht*width,1);
opts.lambda=lambda1;

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
    term3=mu_templ+E_templ*E_templ'*(reshape(recons,[ht*width 1])-mu_templ);
    term3=reshape(term3, [ht width]);
    term3=recons-term3;
    term3=term3.*W;
    term3=norm(term3,'fro')^2;
    ftheta=term1+lam2*term3;
end
function [ans_vec] = grad(I_mat,y_test,theta,sig,E_templ,mu_templ,angles,W)
    I_mat=reshape(I_mat,[l*q,1]);
    y=reshape(y_test,[l*q,1]);
    m=length(y);
    hw=length(theta);
    recons=reshape(theta,[ht,width]);
    alpha_templ=transpose(E_templ)*(recons-mu_templ); % the alpha for term 3
    recons=idct2(recons);
    P=radon(recons,angles);
    M1=phi*psi;% need to modify this-- not correct yet
    M2=W*psi;% need to modify this-- not correct yet
    [W1,~]=size(W);
    ans_vec=zeros(hw,1);
    for val=1:hw
        sum1=0;
        for i=1:m
            term1=y(i)+I_mat(i)*exp(-1*P(i));
            term2=(y(i)-I_mat(i)*exp(-1*P(i)) -2*sig*sig);
            term3=I_mat(i);
            term4=M1(i,val);
            term5=(I_mat(i)*exp(-1*P(i)) + sig*sig);
            sum1=sum1+(term1*term2*term3*term4)/term5;
        end
        sum2=0;
        left_term=W*recons;
        right_term=W*(mu_templ + E_templ*alpha_templ);
        for i=1:W1
            sum2=sum2+(left_term(i)-right_term(i))*M2(i,val);
        end
        ans_vec(val)=sum1+2*lambda2*sum2;
    end
end

function Ftheta=calc_F(theta)
    Ftheta=calc_f(theta);
    Ftheta=Ftheta+lambda1*norm(theta,1);
end