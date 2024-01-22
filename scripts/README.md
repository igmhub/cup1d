# Example of forecast using eBOSS P1D and high-resolution P1D
python forecast.py --timeout 4 --data_label QMLE_Ohio --extra_p1d_label Karacayli2022

# Same, but using Gadget simulation to make mock (instead of model)
python sample_gadget.py --timeout 4 --data_cov_label QMLE_Ohio --extra_p1d_label Karacayli2022

# Old script to add extra columns to Planck chains (might not be updated)
python add_linP_chains.py

