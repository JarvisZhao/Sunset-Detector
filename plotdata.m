function plotdata(X, Y, x1ran, x2ran)
% PLOTDATA - Plot 2D data set
% 

hold on;
ind = find(Y>0);
plot(X(ind,1), X(ind,2), 'ks');
ind = find(Y<0);
plot(X(ind,1), X(ind,2), 'kx');
text(X(:,1)+.2,X(:,2), int2str([1:length(Y)]'));
axis([x1ran x2ran]);
axis xy;
