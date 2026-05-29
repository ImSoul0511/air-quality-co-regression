$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$do_cd = 1;

@default_files = ('report/main.tex');

$clean_ext = 'synctex.gz nav snm vrb bbl run.xml';
