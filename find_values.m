test_case=1;
if test_case==0 % okra
    n=4; % total (n+1) links => n training volumes
    ht=135;
    limitl=146;
    limitr=280;
    limitu=111;
    limitd=245;
    no_slices=123;
elseif test_case==1
    n=3;
    ht=130;
    limitl=11;
    limitr=140;
    limitu=6;
    limitd=135;
    no_slices=100;
else
    n=5;
    ht=140;
    limitl=86;
    limitr=225;
    limitu=96;
    limitd=235;
    no_slices=130;
end

% so, if p=the length of dimension perpendicular to which slices are being
% taken... => we have p sets of 2D slices with size of each set=n (1 slice from each
% template volume and all are in correspondance with each other).
 
width=ht;
% append all the volumes into a vector => x_train
x_train=zeros(ht,width,no_slices,n);
for volume=1:n
    if test_case==0
        arr=['fdk_okra',num2str(volume+3),'.mat'];
    elseif test_case==1
        arr=['fdk_potato',num2str(volume),'.mat'];
    else
        arr=['fdk_sprouts',num2str(volume),'.mat'];
    end
    file_name=join(arr,'');
    s=load(file_name);
    % fieldnames(s)
    if test_case==2
        x_train(:,:,:,volume)=s.volume(limitl:limitr,limitu:limitd,:);
    else
        x_train(:,:,:,volume)=s.FDK(limitl:limitr,limitu:limitd,:);
    end
end
if test_case==0
    arr=['fdk_okra',num2str(3),'.mat'];
elseif test_case==1
    arr=['fdk_potato',num2str(n+1),'.mat'];
else
    arr=['fdk_sprouts',num2str(n+1),'.mat'];
end
file_name=join(arr,'');
s=load(file_name);
if test_case==2
    x_test=s.volume(limitl:limitr,limitu:limitd,:);
else
    x_test=s.FDK(limitl:limitr,limitu:limitd,:);
end

% the number of the slice we are choosing among p sets of slices
if test_case==0
    slice_number=30; 
    normalised_sum=30;
elseif test_case==1
    slice_number=27;
    normalised_sum=50;
else
    slice_number=30; 
    normalised_sum=9;
end

x_train=(x_train/sum(x_train(:,:,slice_number,1),'all'))*normalised_sum;
x_test=(x_test/sum(x_test(:,:,slice_number),'all'))*normalised_sum;
% Here, we assume that slices are along direction with length d3..and
% take the (slice_number)th set of slices...
R=radon(x_train(:,:,1,1),0);
l=length(R);
% [ht,width]=size(x_train(:,:,1,1));

if test_case==0
    q=360;
elseif test_case==1
    q=900;
else
    q=360;
end

y_train=zeros(l,q,n);

mean=zeros(l,q);
mu_templ=zeros(ht,width);

angles=zeros(1,q);
for i=1:q
    angles(i)=179*(i-1)/q;
end

if test_case==0
    I_high=10000;
    sig=0.1;
elseif test_case==1
    I_high=38000;
    sig=0;
else
    I_high=32000;
    sig=0;
end
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
if test_case==0
    no_eigen=3;
elseif test_case==1
    no_eigen=3;
else
    no_eigen=3;
end
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
if test_case==0
    I_low=5000;
elseif test_case==1
    I_low=4000;
else
    I_low=4000;
end
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
if test_case==0
    patch_size=3;
elseif test_case==1
    patch_size=1;
else
    patch_size=3;
end
for j=1:q
    for i=1:l
        low=max([i-patch_size 1]);
        up=min([i+patch_size l]);
        test=hyp(low:up,j);
        [h,p(i,j)]=ztest(test,0,0.5);
    end
end
W_in=iradon((1-p),angles,'linear','Cosine',ht);
W_sq=W_in.*W_in;
W=1./(1+W_sq);
W=rescale(W);

if test_case==0
    p_limit=0.20;
elseif test_case==1
    p_limit=0.20;
else
    p_limit=0.25;
end

I_irr=zeros(l,q);
I_irr(p<p_limit)=I_high;
slice=x_test(:,:,slice_number);
y_test_irr=irradiate_noise(slice,angles,I_irr,sig);
y_test_irr(p>=p_limit)=y_test(p>=p_limit);
I_irr(p>=p_limit)=I_mat_low(p>=p_limit);

if test_case==0
    file_name='okra-values.mat';
elseif test_case==1
    file_name='potato-values.mat';
else
    file_name='sprouts-values.mat';
end

save(file_name,'W', 'mu_templ','E_tmpl','y_test','l','q','angles','ht','width','I_mat_low','sig','normalised_sum','I_irr','y_test_irr');



function proj=irradiate(sample,theta,I,sigma)
    proj=radon(sample,theta);
    proj=exp(-proj);
    proj=I.*proj;
%     proj=poissrnd(proj);
%     proj(proj==0)=1;
    s=size(proj);
%     proj=proj+randn(s)*sigma;
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

