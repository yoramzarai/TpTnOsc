% This function computes s^{-}, s^{+}, s_c^{-} and s_c^{+} of a given set of vectors. 
%
%  Usage: [ s_minus, sc_minus, s_plus, sc_plus ] = compute_sign_variation( x )
%
%  Where: x - a matrix size k by n, treated as k vectors of size n each.
%
%         s_minus - a vector size k, where s_minus(k) is the s^{-} of the vector
%                   x(k,:). Similarly for the other outputs.
%
%
% Yoram Zarai, 5/18/18

% ----------------------------------------------------------------------------------------------

function [ s_minus, sc_minus, s_plus, sc_plus ] = compute_sign_variation( x )
  num_vecs = size(x, 1 );
  s_minus = zeros( num_vecs, 1 );
  sc_minus = zeros( num_vecs, 1 );
  s_plus = zeros( num_vecs, 1 );
  sc_plus = zeros( num_vecs, 1 );
  
  for cnt = 1 : num_vecs
    vec = x( cnt, : );
    % computation of s^{-} and s_c^{-}
    s_minus( cnt ) = compute_sminus( vec );
    sc_minus( cnt ) = s_minus( cnt ) + mod( s_minus( cnt ), 2 );
  
    % computation of s^{+} and s_c^{+}
    zero_indxs = find( vec == 0 );
    if( ~isempty( zero_indxs ) )
      len = length( zero_indxs );
      all_comb = dec2bin(0:(2^len)-1)-'0';
      all_comb( all_comb == 0) = -1;
      max_v = 0;
      temp = vec;
      for indx = 1 : 2^len
        temp( zero_indxs ) = all_comb( indx, : );
        sv = compute_sminus( temp );
        if( sv > max_v )
          max_v = sv;
        end
      end
      s_plus( cnt ) = max_v;
      sc_plus( cnt ) = s_plus( cnt ) + mod( s_plus( cnt ), 2 );
    else
      s_plus( cnt ) = s_minus( cnt );
      sc_plus( cnt ) = sc_minus( cnt );
    end
  
  end

  end
  
% ====================================================================================
  
function s_minus = compute_sminus( vec )
  s_minus = sum( abs( diff( sign( vec( vec ~= 0 ) ) )/2 ) );
  end