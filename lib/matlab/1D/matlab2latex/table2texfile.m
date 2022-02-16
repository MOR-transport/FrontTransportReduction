function latex=table2texfile(input,filename,makeCompleteLatexDocument,option)
%%
%Call:
%   latex=table2texfile(table,filename,makeCompleteLatexDocument,option)
% Input:
%    table       ...table of results generated with table or struct to table
%    filename    ...name of file or path to file
%    makeCompleteLatexDocument    if ==1 then latex begin{document} etc. is
%                                 included
%    option.PERMISSION ... opens the file FILENAME in the
%     mode specified by PERMISSION:
%         'r'     open file for reading
%         'w'     open file for writing; discard existing contents
%         'a'     open or create file for writing; append data to end of file
%         'r+'    open (do not create) file for reading and writing
%         'w+'    open or create file for reading and writing; discard 
%                 existing contents
%         'a+'    open or create file for reading and writing; append data 
%                 to end of file
%         'W'     open file for writing without automatic flushing
%         'A'     open file for appending without automatic flushing
%    input      ... struct containing data and optional fields 
%                 input.data=table;
%                  % Set column labels (use empty string for no label):
%                  input.tableColLabels = {'col1','col2','col3'};
%                  % Set row labels (use empty string for no label):
%                  input.tableRowLabels = {'row1','row2','','row4'};
%                 
%                  % Switch transposing/pivoting your table:
%                 input.transposeTable = 0;
%                 
%                  % Determine whether input.dataFormat is applied column or row based:
%                  input.dataFormatMode = 'column'; % use 'column' or 'row'. if not set 'column' is used
%                 
%
% Output:
%   latex        ...string with data in latex table format
%
%  See also: struct2table, table
%%

if nargin<3 || isempty(makeCompleteLatexDocument)
    makeCompleteLatexDocument=0;
end

if nargin<4 || isempty(option)
    option.PERMISSION='w';
end
% Switch transposing/pivoting your table if needed:
input.transposeTable = 0;

% Switch to generate a complete LaTex document or just a table:
input.makeCompleteLatexDocument =makeCompleteLatexDocument;

% Switch table borders on/off (borders are enabled by default):
input.tableBorders = 0;
input.booktabs = 1;
if ~isfield(input,'tablePlacement')
    input.tablePlacement = 'htp!';
end
% Now call the function to generate LaTex code:
latex = latexTable(input);

% save LaTex code as file
fid=fopen(filename,option.PERMISSION);
[nrows,ncols] = size(latex);
for row = 1:nrows
    fprintf(fid,'%s\n',latex{row,:});
end
fclose(fid);
fprintf('\n Table saved to: %s\n',filename);


end