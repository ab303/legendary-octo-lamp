function []=neural_network(trFile,tFile, layers, units, rounds)
trainingData = dlmread(trFile);
testData = dlmread(tFile);
[nrow, ncollumn] = size(trainingData);
[nTrow, ~] = size(testData);
ncollumn = ncollumn - 1;
t = trainingData(:,end);
l = layers;
cvalues = unique(t);
[nclass, ~] = size(cvalues);
U = ncollumn+1 + nclass + (layers-2)*units + l - 1;
tn = zeros(nrow, U);
sum1 = zeros(1,1);
b = 1;
lR = 1;
for k = 1:nrow
    for i = U-nclass+1:U
        if t(k,1)==cvalues(b,1)
            tn(k,i)=1;
            b=b+1;
        else
            tn(k,i)=0;
            b=b+1;
        end
    end
    b = 1;
end

%initialisation
z = ones(U,1);
w = (0.5+0.5).*rand(U,U)-0.5;
xn = ones(nrow,1);
y = trainingData(:,1:end-1);
[m,~] = max(y);
y = y/m(1,1);
xn = [xn, y];
a = zeros(U,1);
delta = zeros(U,1);



%stopping according to number of rounds
for r=1:rounds
%iterate over all objects
for n=1:nrow
    %initialise the input units
    for j=1:ncollumn+1
        z(j,1) = xn(n,j);
    end
    if l > 2
        for k=2:l
            %calculating output z
            if k==2
                for j=ncollumn+2:ncollumn+1+units
                    for i=1:ncollumn+1
                        a(j,1) = w(j,i)*z(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    z(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
            elseif k==l
                
                for j=U-nclass+1:U
                    sum1= w(j,end)*z(1,1);
                    for i=ncollumn+2+(k-3)*units:U-nclass
                        a(j,1) = w(j,i)*z(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    z(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
            else
                
                for j=ncollumn+2+(k-2)*units:ncollumn+1+(k-1)*units
                    sum1= w(j,end)*z(1,1);
                    for i=ncollumn+2+(k-3)*units:ncollumn+2+(k-2)*units
                        a(j,1) = w(j,i)*z(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    z(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
               
            end
        end
    elseif l == 2 %update this
        for j=U-nclass+1:U
            for i=1:ncollumn+1
                 a(j,1) = w(j,i)*z(i,1);
                 sum1 = sum1 + a(j,1);
            end
            z(j,1) = 1./(1+exp(-sum1));
            sum1 = 0;
        end
    end
%end of output function
    if l==2
        for j=U-nclass+1:U
            delta(j,1)=(z(j,1)-tn(n,j))*z(j)*(1-z(j));
            for i = 1:ncollumn+1
                w(j,i) = w(j,i) - lR * delta(j,1)* z(i,1);
            end
            w(j,end) = w(j,end) - lR * delta(j,1) * z(1,1);
        end
    else
        for j=U-nclass+1:U
            delta(j,1)=(z(j,1)-tn(n,j))*z(j)*(1-z(j));
            for i = U-nclass-units+1:U-nclass
                w(j,i) = w(j,i) - lR * delta(j,1)* z(i,1);
            end
            w(j,end) = w(j,end) - lR * delta(j,1) * z(1,1);
        end
    end
    sum1 = 0;
    k = l-1;
    while k~=1
       
        if k==2
            if k+1==l
                for j=ncollumn+2:ncollumn+1+units
                    for u=U-nclass+1:U
                        a(j,1) = delta(u,1)*w(u,j);
                        sum1 = sum1 + a(j,1);
                    end
                    delta(j,1) = sum1*z(j)*(1-z(j));
                    sum1=0;
                    for i = 1:ncollumn+1
                        w(j,i) = w(j,i) - lR * delta(j,1) * z(i,1);
                    
                    end
                end
                k=k-1;
            else
                
                for j=ncollumn+2:ncollumn+1+units
               
                    for u=ncollumn+1+(k-1)*units+1:ncollumn+1+(k)*units
                       a(j,1) = delta(u,1)*w(u,j);
                       sum1 = sum1 + a(j,1);
                    end
             
                    delta(j,1) = sum1*z(j)*(1-z(j));
                    sum1=0;
                    for i = 1:ncollumn+1
                        w(j,i) = w(j,i) - lR * delta(j,1) * z(i,1);
                    
                    end
                end
                k=k-1;
            end
        else
            if k==l-1
                for j=ncollumn+2+(k-2)*units:U-nclass
                    for u = U-nclass+1:U
                        a(j,1) = delta(u,1)*w(u,j);
                        sum1 = sum1 + a(j,1);
                    end
                    delta(j,1) = sum1*z(j)*(1-z(j));
                    sum1=0;
                    for i=ncollumn+1+(k-3)*units+1:U-nclass-units
                        w(j,i) = w(j,i) - lR * delta(j,1) * z(i,1);
                    end
                end
            else
                if k+1==l-1
                    for j=ncollumn+2+(k-2)*units:U-nclass-units
                        for u = U-nclass-units+1:U-nclass
                            a(j,1) = delta(u,1)*w(u,j);
                            sum1 = sum1 + a(j,1);
                        end
                        delta(j,1) = sum1*z(j)*(1-z(j));
                        sum1=0;
                        for i = ncollumn+1+(k-3)*units+1:ncollumn+1+(k-2)*units
                            w(j,i) = w(j,i) - lR * delta(j,1) * z(i,1);
                        end
                    end
                else
                    for j=ncollumn+2+(k-2)*units:ncollumn+1+(k-1)*units
                        for u = ncollumn+1+(k-1)*units+1:ncollumn+1+(k)*units
                            a(j,1) = delta(u,1)*w(u,j);
                            sum1 = sum1 + a(j,1);
                        end
                        delta(j,1) = sum1*z(j)*(1-z(j));
                        sum1=0;
                        for i = ncollumn+1+(k-3)*units+1:ncollumn+1+(k-2)*units
                            w(j,i) = w(j,i) - lR * delta(j,1) * z(i,1);
                        end
                    end
                end
            end
            k=k-1;
        end
       
    end 
    
    
end
lR=lR*0.98; %incrementing learning rate after each round
end
% end of training

zT = ones(U,1);
xnT = ones(nTrow,1);
f = testData(:,1:end-1);
tT = testData(:,end);
[mT,~] = max(f);
f = f/mT(1,1);
xnT = [xnT, f];
sum1 = 0;
acc = zeros(nTrow,1);

for n=1:nTrow
    %initialise the input units
    for j=1:ncollumn+1
        zT(j,1) = xnT(n,j);
    end
    if l > 2
        for k=2:l
            %calculating output z
            if k==2
                for j=ncollumn+2:ncollumn+1+units
                    for i=1:ncollumn+1
                        a(j,1) = w(j,i)*zT(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    zT(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
            elseif k==l
                
                for j=U-nclass+1:U
                    sum1= w(j,end)*zT(1,1);
                    for i=ncollumn+2+(k-3)*units:U-nclass
                        a(j,1) = w(j,i)*zT(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    zT(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
            else
                
                for j=ncollumn+2+(k-2)*units:ncollumn+1+(k-1)*units
                    sum1= w(j,end)*zT(1,1);
                    for i=ncollumn+2+(k-3)*units:ncollumn+2+(k-2)*units
                        a(j,1) = w(j,i)*zT(i,1);
                        sum1 = sum1 + a(j,1);
                    end
                    zT(j,1) = 1./(1+exp(-sum1));
                    sum1 = 0;
                end
               
            end
        end
    elseif l == 2 
        for j=U-nclass+1:U
            for i=1:ncollumn+1
                 a(j,1) = w(j,i)*zT(i,1);
                 sum1 = sum1 + a(j,1);
            end
            zT(j,1) = 1./(1+exp(-sum1));
            sum1 = 0;
        end
    end
    
    %classification
    o = zT(U-nclass+1:U,1);
    [~, idx] = max(o);
    minC = min(cvalues);
    if minC==0
        class = idx-1;
    else
        class = idx;
    end
    if tT(n,1)==class
        acc(n,1) = 1;
    else
        acc(n,1) = 0;
    end
    fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f \n', n-1, class, tT(n,1), acc(n,1));
end
fprintf('classification accuracy=%6.4f\n', sum(acc)/nTrow);
end