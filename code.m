n=4; % currently considering okra dataset.. total 5 links => 4 training volumes
% so, if p=the length of dimension perpendicular to which slices are being
% taken... => we have p sets of 2D slices with size of each set=n (1 slice from each
% template volume and all are in correspondance with each other).
 
% append all the volumes into a vector => x_train
x_train=zeros(338,338,123,n);
for volume=1:n
    arr=['fdk',num2str(volume+2),'.mat'];
    file_name=join(arr,'');
    s=load(file_name);
    % fieldnames(s)
    x_train(:,:,:,volume)=s.FDK(:,:,:);
end
arr=['fdk',num2str(n+3),'.mat'];
file_name=join(arr,'');
s=load(file_name);
x_test=s.FDK(:,:,:);

% the number of the slice we are choosing among p sets of slices
slice_number=1; 
% Here, we assume that slices are along direction with length d3..and
% take the (slice_number)th set of slices...
R=radon(x_train(:,:,1,1),0);
l=length(R);
[ht,width]=size(x_train(:,:,1,1));

q=360;

y_train=zeros(l,q,n);

mean=zeros(l,q);
mu_templ=zeros(ht,width);

angles=zeros(1,q);
for i=1:q
    angles(i)=179*(i-1)/q;
end

I_high=1000;
I_mat=ones(l,q)*I_high;
% mean and projections of training templates.
for i=1:n
    slice=x_train(:,:,slice_number,i);
    mu_templ=mu_templ+slice;
    y_train(:,:,i)=irradiate(slice,angles,I_mat,0);
    mean(:,:)=mean(:,:)+y_train(:,:,i);
end 
mu_templ=mu_templ./n;
mu_templ=reshape(mu_templ,[ht*width 1]);
mean=mean./n;
%mean(:,:)

% Covariance matrices.
Cov=zeros(l,l,q);
Cov_templ=zeros(ht*width,ht*Width);
for angle=1:q
    for i=1:n
        tmp=y_train(:,angle,i)-mean(:,angle);
        Cov(:,:,angle)=Cov(:,:,angle)+tmp*transpose(tmp);
    end
end
for i=1:n
    slice=x_train(:,:,slice_number,i);
    slice=reshape(slice,[ht*width 1]);
    tmp=slice-mu_templ;
    Cov_templ=Cov_templ+tmp*tmp';
end
% Cov(:,:,:)

% Eigen spaces.
E=zeros(l,l,q);
for angle=1:q
    [V,~]=eig(Cov(:,:,angle));
    E(:,:,angle)=E(:,:,angle)+V;
end
[E_tmpl,~]=eig(Cov_templ);
% E(:,:,:)

% y -> projections of the test template.
y_test=zeros(l,q);
I_low=10;
I_mat_low=ones(l,q)*I_low;
slice=x_test(:,:,slice_number);
y_test=irradiate(slice,theta,I_mat_low,sig);
% for angle=1:q
%     y_test(:,angle)=radon(slice,theta(angle));
% end
% y_test(:,:)

% coefficients of y_test along eigen vectors
% alpha=zeros(l,q);
% for angle=1:q
%     alpha(:,angle)=transpose(E(:,:,angle))*(y_test(:,angle)-mean(:,angle));
% end
% alpha(:,:)

% resultant projections
y_p=zeros(l,q);
for angle=1:q
   y_p(:,angle)=mean(:,angle)+ E(:,:,angle)*(transpose(E(:,:,angle))*(y_test(:,angle)-mean(:,angle)));
end
% y_p(:,:)


% general anscombe transform
hyp=sqrt(y+(3/8)+sig*sig)-sqrt(y_p+(3/8)+sig*sig);

% result of z test
p=zeros(l,q);
for j=1:q
    for i=1:l
        low=max([i-5 1]);
        up=min([i+5 l]);
        test=hyp(low:up);
        [h,p(i,j)]=ztest(test,0,0.5);
    end
end
W_in=iradon(p,theta,'linear','Cosine');
W_sq=W_in.*W_in;
W=1./(1+W_sq);
W=rescale(W);






% column_matrix_x=zeros(d2*d3,n);
% column_matrix_y=zeros(d3*d1,n);
% column_matrix_z=zeros(d1*d2,n);
% u_x=zeros(d2*d3);
% u_y=zeros(d3*d1);
% u_z=zeros(d1*d2);
% for i=1:n
%     % along a given direction.. how to extract the volume projection?
%     column_matrix_x(:,i)=reshape(data(1,:,:,i),[d2*d3,1]);
%     u_x=u_x+column_matrix_x(:,i);
%     column_matrix_y(:,i)=reshape(data(:,1,:,i),[d3*d1,1]);
%     u_y=u_y+column_matrix_y(:,i);
%     column_matrix_z(:,i)=reshape(data(:,:,1,i),[d1*d2,1]);
%     u_z=u_z+column_matrix_z(:,i);
% end
% u_x=u_x./n;
% u_y=u_y./n;
% u_z=u_z./n;
% % the dimensions of different eigenspaces will be different..how to combine
% % them? slice - volume dilemma
% [V_x,D_x]=eig(column_matrix_x);
% [V_y,D_y]=eig(column_matrix_y);
% [V_z,D_z]=eig(column_matrix_z);
% % y must be given in the data
% alpha_x=transpose(V_x)*(y_x-u_x);
% alpha_y=transpose(V_y)*(y_y-u_y);
% alpha_z=transpose(V_z)*(y_z-u_z);
% yp_x=u_x+V_x*alpha_x;
% yp_y=u_y+V_y*alpha_y;
% yp_z=u_z+V_z*alpha_z;

function proj=irradiate(sample,theta,I,sigma)
    proj=radon(sample,theta);
    proj=exp(-proj);
    proj=I.*proj;
    proj=poissrnd(proj);
    s=size(proj);
    proj=proj+randn(s)*sigma;
end

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
function [ans_vec] = grad(I0,y,M,theta,sig)
    m=length(y);
    recons=reshape(theta,[ht,width]);
    recons=idct2(recons);
    P=radon(recons,angles);
    ans_vec=zeros(q,1);
    for val=1:q
        sum=0;
        for i=1:m
            term1=y(i)+I0*exp(-1*P(i));
            term2=(y(i)-I0*exp(-1*P(i)) -2*sig*sig);
            term3=I0;
            term4=M(i,val);
            term5=(I0*exp(-1*P(i)) + sig*sig);
            sum=sum+term1*term2*term3*term4/term5;
        end
        ans_vec(val)=sum;
    end
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

end

    
    