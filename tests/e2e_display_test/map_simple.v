// Techmap for simple cells to AIGPDK gates
module \$_NOT_ (A, Y);
  input A;
  output Y;
  INV _TECHMAP_REPLACE_ (.A(A), .Y(Y));
endmodule

module \$_AND_ (A, B, Y);
  input A, B;
  output Y;
  AND2_11_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
endmodule

module \$_OR_ (A, B, Y);
  input A, B;
  output Y;
  wire _n;
  INV _inv_a (.A(A), .Y(_n_a));
  INV _inv_b (.A(B), .Y(_n_b));
  AND2_00_0 _and (.A(_n_a), .B(_n_b), .Y(_n));
  INV _inv_out (.A(_n), .Y(Y));
endmodule
