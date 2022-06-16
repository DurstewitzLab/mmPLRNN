function [iFRtrafo_,C__,Tmtx_,Inp_]=cutkCVE(k,iFRtrafo,C_,Tmtx,Inp)
%Philine Bommer 04.11.19
%cuts according trial for kfold-CVE from categorical and gaussian data
      
      iFRtrafo(:,k)=[];
      iFRtrafo_=iFRtrafo;
      C_(:,k)=[];
      C__=C_;
      Tmtx(:,k)=[];
      Tmtx_=Tmtx;
      Inp(:,k)=[];
      Inp_=Inp;

%     for i=1:length(iFRtrafo)
%         if i==k
%         else
%             iFRtrafo_=cell(1,(length(iFRtrafo)-1));
%             C__=cell(1,(length(C_)-1));
%         end
%     end


end