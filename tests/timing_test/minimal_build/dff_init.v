// Override the DFF UDP to initialize Q=0 (matching CXXRTL/GEM behavior)
`ifdef SKY130_FD_SC_HD__UDP_DFF_P_V
`else
`define SKY130_FD_SC_HD__UDP_DFF_P_V
primitive sky130_fd_sc_hd__udp_dff$P (
    Q  ,
    D  ,
    CLK
);
    output Q  ;
    input  D  ;
    input  CLK;
    reg Q;
    initial Q = 1'b0;
    table
     //  D  CLK  :  Qt : Qt+1
         1  (01) :  ?  :  1    ;
         0  (01) :  ?  :  0    ;
         1  (x1) :  1  :  1    ;
         0  (x1) :  0  :  0    ;
         1  (0x) :  1  :  1    ;
         0  (0x) :  0  :  0    ;
         ?  (1x) :  ?  :  -    ;
         ?  (?0) :  ?  :  -    ;
         *   ?   :  ?  :  -    ;
    endtable
endprimitive
`endif

// Also override the NSR version (DFF with async set/reset)
`ifdef SKY130_FD_SC_HD__UDP_DFF_NSR_V
`else
`define SKY130_FD_SC_HD__UDP_DFF_NSR_V
primitive sky130_fd_sc_hd__udp_dff$NSR (
    Q    ,
    SET  ,
    RESET,
    CLK  ,
    D
);
    output Q    ;
    input  SET  ;
    input  RESET;
    input  CLK  ;
    input  D    ;
    reg Q;
    initial Q = 1'b0;
    table
     //  S  R  C   D  :  Qt : Qt+1
         0  1  ?   ?  :  ?  :  0    ; // Reset
         1  ?  ?   ?  :  ?  :  1    ; // Set (dominates)
         0  0  (01) 0 :  ?  :  0    ; // clocked data
         0  0  (01) 1 :  ?  :  1    ;
         0  0  (x1) 0 :  0  :  0    ; // reducing pessimism
         0  0  (x1) 1 :  1  :  1    ;
         0  0  (0x) 0 :  0  :  0    ;
         0  0  (0x) 1 :  1  :  1    ;
         0  0  ?    *  :  ?  :  -   ; // data changes - loss of info
         0  0  (?0) ?  :  ?  :  -   ; // falling/no-edge CLK
         0  0  (1x) ?  :  ?  :  -   ;
    endtable
endprimitive
`endif

// Override PR version (DFF with async preset/reset)
`ifdef SKY130_FD_SC_HD__UDP_DFF_PR_V
`else
`define SKY130_FD_SC_HD__UDP_DFF_PR_V
primitive sky130_fd_sc_hd__udp_dff$PR (
    Q    ,
    D    ,
    CLK  ,
    RESET
);
    output Q    ;
    input  D    ;
    input  CLK  ;
    input  RESET;
    reg Q;
    initial Q = 1'b0;
    table
     //  D  CLK    R  :  Qt : Qt+1
         ?  ?      1  :  ?  :  0    ; // Reset
         0  (01)   0  :  ?  :  0    ;
         1  (01)   0  :  ?  :  1    ;
         0  (x1)   0  :  0  :  0    ;
         1  (x1)   0  :  1  :  1    ;
         0  (0x)   0  :  0  :  0    ;
         1  (0x)   0  :  1  :  1    ;
         ?  (1x)   0  :  ?  :  -    ;
         ?  (?0)   0  :  ?  :  -    ;
         *  ?      0  :  ?  :  -    ;
         ?  ?      (?0) :  ?  :  -  ;
    endtable
endprimitive
`endif
