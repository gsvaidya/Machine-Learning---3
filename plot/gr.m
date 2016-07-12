X = importdata('gamma.txt');
Y = importdata('C.txt');
Z = importdata('acc1.txt');
 
Tri = delaunay(X,Y);
C = Z;
trisurf(Tri,X,Y,Z,C);
set(gca,'XScale','log');
set(gca,'YScale','log');
X = importdata('degree.txt');
Y = importdata('C.txt');
Z = importdata('acc2.txt');
Tri = delaunay(X,Y);
C = Z;
trisurf(Tri,X,Y,C);
set(gca,'YScale','log');
colormap copper
