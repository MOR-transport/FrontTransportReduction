% dispone.m
%
% Usage: dispone(val, dval, dig) 
%   val and val must have length one
%
% dispone gives a string with the values and errors in the form
% val(dval), where dval is truncated to dig digits
%
%****************************************************

function [output] = dispone(val, dval, dig)

   if nargin < 3
      dig = 2;
   end
   
   n=length(val);
   
   if n~=1
       error('[dispone] Unexpected length of val!');
       return
   end
   
   if n~=length(dval)
       error('[dispone] val and dval must have the same length!');
       return
   end

   if dval==0
      output = num2str(val);

   elseif dval<10
      location = floor(log10(dval));
      append_err = ['(' num2str(round(dval*10^(-location+dig-1))) ')'];
      output = [num2str(val,['%.' num2str(-location+dig-1) 'f']) append_err];

   else
      digits = max(0,ceil(log10(dval))-dig);
      err    = round(dval/10^digits)*10^digits;
      val    = round(val/10^digits)*10^digits;
      output = [num2str(val) '(' num2str(err) ')'];
       
   end

end






