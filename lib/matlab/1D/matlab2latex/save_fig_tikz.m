function save_fig_tikz(fname,figurehandle,extraopt)
%
% call: save_fig_tikz(fname,figurehandle,extraopt)
%
%
%
if nargin<2 || isempty(figurehandle)
    figurehandle=gcf;
end

if nargin<3 || isempty(extraopt)
    % extraopt='xticklabel style={/pgf/number format/fixed},yticklabel style={/pgf/number format/fixed}';
    extraopt='';
    h=gca;
    objs=h.findobj;
    if isa(objs(end),'matlab.graphics.primitive.Image')
%        extraopt=[extraopt, 'xticklabels={',char(join(string(h.XTickLabel(:,:)),',')),'}'];
 %       extraopt=[extraopt, ', yticklabels={',char(join(string(h.YTickLabel(:,:)),',')),'}'];
    else
        extraopt='scaled ticks=false, xticklabel style={/pgf/number format/fixed},yticklabel style={/pgf/number format/fixed}';
    end
end
fnametikz = [char(fname) , '.tikz.tex'];
matlab2tikz('height', '\figureheight', 'width', '\figurewidth',  ...
    'showInfo',false,'figurehandle',figurehandle,'floatFormat','%.10f', fnametikz);
print(figurehandle,[fname],'-dpng')
%print(figurehandle,[fname],'-dpdf')
end