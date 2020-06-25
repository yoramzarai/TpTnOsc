% This script contains examples for using TpTnOsc functions from Matlab.
%
% Note: Make sure in Matlab to run "pyversion <path to python>" (or in 
%       Matlab2019b "pyenv <path to python>"). 
%
%
% Yoram Zarai, 7/10/19

% ------------------------------------------------------------------------------------
clear all;
close all;

% Computing the p'th multiplicative compound (MC) of a matrix
% ===========================================================
A = [ 1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
p = 2;
% this python function computes the p'th multiplicative compound of a matrix
ret = py.TpTnOsc.utils.compute_MC_matrix( py.numpy.array(A), int8(p) );
mc_mat = double( ret{1} ); % the p'th MC matrix
indxs = int8( ret{2} ); % the lexicography indexes of the rows and columns in the MC matrix

% computing the MC matrix of the matrix L(n, i, q) (the L matrix in the EB factorization)
% =======================================================================================
n = 4;
i = 3;
q = 5;
Lmat = py.TpTnOsc.utils.L(int8(n), int8(i), int8(q));
reta = py.TpTnOsc.utils.compute_MC_matrix( Lmat, int8(p) );
Lmat_mc = double( reta{1} );

% Generating a matrix from its EB factorization parameters
% =========================================================
valsL = py.numpy.array([2 3 0]);  % the L matrix parameters (l_1 to l_k)
valsD = py.numpy.array([1 1 1]);  % the diagonal of D (d_1 to d_n)
valsU = py.numpy.array([5 10 0]); % the U matrix parameters (u_1 to u_k)
% this python function computes the I-TN matrix from its EB factorization parameters
ret1 = py.TpTnOsc.utils.compute_matrix_from_EB_factorization( valsL, valsD, valsU );
fprintf( 1, 'Generated matrix = \n' );
double(ret1)

% Computing the EB factorization of a ITN matrix
% ================================================
B = [ 3 1 0 0; 1 4 1 0.2; 0.1 1 5 3; 0 0 2 7 ];
% this python function computes the EB factorization of an I-TN matrix
ret2 = py.TpTnOsc.utils.EB_factorization_ITN( py.numpy.array( B ) );
fprintf( 1, 'Following are the EB factorization of:\n' ); B
for cnt = 1 : length( ret2{1} ) % the L matrices
  fprintf( 1, 'L(%d)=\n', cnt ); double( ret2{1}{cnt} )
end
fprintf( 1, 'D = \n' ); double( ret2{2} )
for cnt = 1 : length( ret2{3} ) % the U matrices
  fprintf( 1, 'U(%d)=\n', cnt ); double( ret2{3}{cnt} )
end


% Display the planar graph given the EB factorization above
% =========================================================
if(0)
valsL = ret2{5};
valsU = ret2{6};
d = diag(double(ret2{2}));
fig_fn = './eb_planar.png';
fig_size = [16, 6]; % matplotlib plot figure size
py.matplotlib.pyplot.switch_backend('Agg');
fax = py.matplotlib.pyplot.subplots(pyargs('figsize',int8(fig_size)));
py.TpTnOsc.utils.draw_EB_factorization_ITN( valsL, py.numpy.array(d), valsU, fax{2} ); 
py.matplotlib.pyplot.savefig(fig_fn)
fprintf( 1, 'EB planar diagram saved in file %s\n', fig_fn );
fg = imread( fig_fn );
imshow(fg);
end

% testing if a given matrix is TP/OSC/I-TN/TN
% ===========================================
%A = [ 1 2 2.1; 1 3 9; 1 4 16];
A = [ 2 1 0; 1 2 1; 0 1 2 ];
tol = 1e-9; % tolerance for determinant > tol

isOSC = py.TpTnOsc.utils.is_OSC( py.numpy.array(A), tol );
isTP = py.TpTnOsc.utils.is_TP( py.numpy.array(A), tol );
isTN = py.TpTnOsc.utils.is_TN( py.numpy.array(A) );
isITN = py.TpTnOsc.utils.is_ITN( py.numpy.array(A), tol );
if( isTP )
  fprintf(1, 'A is TP.\n' );
elseif( isOSC )
  fprintf(1, 'A is osc.\n' );
elseif( isITN )
  fprintf( 'A is I-TN.\n' );
elseif( isTN )
    fprintf(1, 'A is TN.\n' );
else
    fprintf(1, 'A is not TN.\n' );
end

% generating an oscillatory matrix
% ================================
% here we use Theorem 2.6.6 in the book Totally Nonnegative matrices
% this is the case n=4, so k=n*(n-1)/2=6
valsL = py.numpy.array([1 1 1 0 0 0]);
valsD = py.numpy.array([1 1 1 1]);
valsU = py.numpy.array([0 1 1 1 0 0]);
B = py.TpTnOsc.utils.compute_matrix_from_EB_factorization( valsL, valsD, valsU );
fprintf(1, 'B = \n' ); double(B)
isOSC = py.TpTnOsc.utils.is_OSC( B, tol );
if( isOSC )
  fprintf( 1, 'B is oscillatory.\n' );
else
  fprintf( 1, 'ERROR - B is not oscillatory!!\n' );
end

% computing the exponent of the oscillatory matrix
% ================================================
if( isOSC )
  r = py.TpTnOsc.utils.osc_exp(B, tol);
  fprintf( 1, 'r(B) = %d.\n', r );
  %B_mat = double(B);
  %BB_mat = py.TpTnOsc.utils.compute_MC_matrix( py.numpy.array(B_mat*B_mat), int8(3) );
  %fprintf( 1, '(B^2)^(3)=\n');
  %double(BB_mat{1})
end;


% number of sign variations
% ===========================
if( 0 )
len = 100;
n = 9;
num_err = 0;
for cnt = 1 : len
  v = randn(1,n);
  v(abs(v)<0.3)=0;
  % sign vatiation Matlab function
  [ s_minus, sc_minus, s_plus, sc_plus ] = compute_sign_variation( v );

  % python version
  ps_minus = double(py.TpTnOsc.utils.s_minus(py.numpy.array(v)));
  psc_minus = double(py.TpTnOsc.utils.sc_minus(py.numpy.array(v)));
  ps_plus = double(py.TpTnOsc.utils.s_plus(py.numpy.array(v)));
  psc_plus = double(py.TpTnOsc.utils.sc_plus(py.numpy.array(v)));
  fprintf( 1, 'v=[%s], s- = %d,%d, s+ = %d,%d, sc- = %d,%d, sc+ = %d,%d ',...
           sprintf('%1.2g ', v ),  s_minus, ps_minus, s_plus, ps_plus, sc_minus, psc_minus, sc_plus, psc_plus );
  if( (s_minus==ps_minus) && (s_plus==ps_plus) && (sc_minus==psc_minus) && (sc_plus==psc_plus) )
    fprintf( 1, ' - OK.\n' );
  else
    fprintf( 1, ' - ERROR !!.\n' );
    num_err = num_err + 1;
  end
end
fprintf( 1, 'Total of %d errors.\n', num_err );
end % if(0)