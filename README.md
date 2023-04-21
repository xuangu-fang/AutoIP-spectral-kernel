

4/20

- classical Horseshoe - will lead to very small weight, then near-zero kernel
    - solution: only use local scaler + softmax + block-diag (ensire K is diag)
    - works well on low freq ( 50 scale + sin(pi x) ), but not on x2 + sinx 
        - always result in all average weights (1/Q)

    - to be test:
        - full cov +only local scaler + softmax? 

- whiting trick (not use log-det term in loss)
    - help convergence on  low freq ( 50 scale + sin(pi x) ),
    - not help (even hard) other case

- real whiting trick + VI