n=4; % currently considering okra dataset.. total 5 links => 4 training volumes
% so, if p=the length of dimension perpendicular to which slices are being
% taken... => we have p sets of 2D slices with size of each set=n (1 slice from each
% template volume and all are in correspondance with each other).
 
% append all the volumes into a vector => x_train
x_train=zeros(135,135,123,n);
for volume=1:n
    arr=['fdk',num2str(volume+2),'.mat'];
    file_name=join(arr,'');
    s=load(file_name);
    % fieldnames(s)
    x_train(:,:,:,volume)=s.FDK(146:280,111:245,:);
end
arr=['fdk',num2str(n+3),'.mat'];
file_name=join(arr,'');
s=load(file_name);
x_test=s.FDK(146:280,111:245,:);

% the number of the slice we are choosing among p sets of slices
slice_number=30; 
normalised_sum=50;
x_train=(x_train/sum(x_train(:,:,slice_number,1),'all'))*normalised_sum;
x_test=(x_test/sum(x_test(:,:,slice_number),'all'))*normalised_sum;
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

I_high=10000;
sig=0;
I_mat=ones(l,q)*I_high;
% mean and projections of training templates.
for i=1:n
    slice=x_train(:,:,slice_number,i);
    mu_templ=mu_templ+slice;
    y_train(:,:,i)=irradiate(slice,angles,I_mat,sig);
    mean(:,:)=mean(:,:)+y_train(:,:,i);
end 
mu_templ=mu_templ./n;
mu_templ=reshape(mu_templ,[ht*width 1]);% mu_templ -> for slices of the volumes
mean=mean./n; % mean -> for projections
%mean(:,:)

% Covariance matrices.
Cov=zeros(l,l,q);
Cov_templ=zeros(ht*width,ht*width);
for angle=1:q
    for j=1:n
        tmp=y_train(:,angle,j)-mean(:,angle);
        Cov(:,:,angle)=Cov(:,:,angle)+tmp*transpose(tmp);
    end
end
for i=1:n
    slice=x_train(:,:,slice_number,i);
    slice=reshape(slice,[ht*width 1]);
    tmp=slice-mu_templ;
    Cov_templ=Cov_templ+tmp*tmp';
end
Cov=Cov/n;
Cov_templ=Cov_templ/n;
% Cov(:,:,:)
no_eigen=3;
E=zeros(l,no_eigen,q);
% Eigen spaces.
for angle=1:q
    [V,~]=eigs(Cov(:,:,angle),no_eigen);
    E(:,:,angle)=V;
end
clear Cov;
[E_tmpl,~]=eigs(Cov_templ,no_eigen);
clear Cov_templ;
% E(:,:,:)

% y -> projections of the test template.
y_test=zeros(l,q);
I_low=1000;
I_mat_low=ones(l,q)*I_low;
slice=x_test(:,:,slice_number);

y_test=irradiate_noise(slice,angles,I_mat_low,sig);
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
% mean=mean./I_mat;
% mean=mean.*I_mat_low;
for angle=1:q
   y_p(:,angle)=max(mean(:,angle)+ E(:,:,angle)*(transpose(E(:,:,angle))*(y_test(:,angle)-mean(:,angle))),0); % mu + V*alpha
end
% y_p(:,:)


% general anscombe transform
hyp=sqrt(y_test*I_low+(3/8)+sig*sig)-sqrt(y_p*I_low+(3/8)+sig*sig);

% result of z test
p=zeros(l,q);
patch_size=3;
for j=1:q
    for i=1:l
        low=max([i-patch_size 1]);
        up=min([i+patch_size l]);
        test=hyp(low:up,j);
        [h,p(i,j)]=ztest(test,0,0.5);
    end
end
W_in=iradon(p,angles,'linear','Cosine');
W_sq=W_in.*W_in;
W=1./(1+W_sq);
W=rescale(W);

save('okra-values.mat','W', 'mu_templ','E_tmpl','y_test','l','q','angles','ht','width','I_mat_low','sig');



function proj=irradiate(sample,theta,I,sigma)
    proj=radon(sample,theta);
    proj=exp(-proj);
    proj=I.*proj;
%     proj=poissrnd(proj);
%     proj(proj==0)=1;
    s=size(proj);
    proj=proj+randn(s)*sigma;
    proj=max(proj,0);
    proj=proj./I;
%     proj=-log(proj);
end

function proj=irradiate_noise(sample,theta,I,sigma)
    proj=radon(sample,theta);
    proj=exp(-proj);
    proj=I.*proj;
    proj=poissrnd(proj);
%     proj(proj==0)=1;
    s=size(proj);
    proj=proj+randn(s)*sigma;
    proj=max(proj,0);
    proj=proj./I;
%     proj=-log(proj);
end

