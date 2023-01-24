
This folder contains the aggregated ready to use datasets with joint data from the RBKP, RSEG, BKPF and BSEG tables.
While the data was processed as discussed in Section 4 of the paper, we additionally add 5 metadata columns to each dataset. The right-most 5 columns of each dataset represent the label of the transaction (fraud-type / no fraud), both  of its identifying values within the BSEG table (document number and position), a general description of its transaction type according to the paper (Invoice / Credit / G/L Account Posting / Material Receipt / Material Withdrawl), and the real time of recording within the ERPsim simulation to be able to maintain the temporal order of the data.

As the ERPsim game was conducted in German, we supply an additional column_information.csv file that contains a mapping to the official English SAP column descriptions. The file additionally contains information on the data type of each column (categorical / numerical) and marks the mentioned metadata columns.

Full ERP data available at https://professor-x.de/erp-fraud-data
[Tritscher, Julian, et al. "Open ERP System Data For Occupational Fraud Detection." arXiv preprint arXiv:2206.04460 (2022).]
