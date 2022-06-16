function [ C ]=Classification_3dim(X)
%% Function simulating class labels based on 1 dim spatial inforamtion:
    %dim: specifies dimension of classification
T=size(X,2);
Ntraj=size(X,3);
C=zeros(8,T,Ntraj);
%thresh(vector): specifies class boundaries based on the mean of each state
thresh=[mean(mean(X(1,:,:))),mean(mean(X(2,:,:))),mean(mean(X(3,:,:)))];

    for i =1:Ntraj
        x=X(1,:,i); % x-component of the lorenz trajectory
        y=X(2,:,i);
        z=X(3,:,i);
        preclass1=sign(x-thresh(1));
        preclass2=sign(y-thresh(2));
        preclass3=sign(z-thresh(3));
        %set up categorical vectorsbased on 8 subspaces:
        for c=1:T
        if preclass1(1,c)==1
            if preclass2(1,c)==1
                if preclass3(1,c)==1
                    C(1,c,i)=1;  
                else
                    C(2,c,i)=1;
                end
            else
                if preclass3(1,c)==1
                    C(3,c,i)=1;
                else
                    C(4,c,i)=1;
                end
            end
        else
            if preclass2(1,c)==1
                if preclass3(1,c)==1
                    C(5,c,i)=1;
                else
                    C(6,c,i)=1;
                end
            else
                if preclass3(1,c)==1
                    C(7,c,i)=1;
                else
                    C(8,c,i)=1;
                end
            end
        end
        end

        if sum(sum(C(:,:,i)))/T==1
            disp('correct labeling')
        else 
            disp('wrong labeling or zeros passage')
            break;
        end
        
    end
        

end