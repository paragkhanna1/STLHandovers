#Extract STL-model
Thesis: Automated Control of Human-Robot Handovers using Data-driven STL Modeling of Human-Human Handovers, 2023
Author: Jonathan Fredberg, jfredb@kth.se

Code based on 
S. Jha, A. Tiwari, S. A. Seshia, T. Sahai, and N. Shankar, “TeLEx: Passive STL learning using only positive examples,” in Runtime Verification: 17th International Conference, RV 2017, Seattle, WA, USA, September 13-16, 2017, Proceedings. Springer, 2017, pp. 208–224.
https://github.com/susmitjha/TeLEX

run successfully in python 2.7.13
Python packages: numpy, scipy, pandas, parsimonious, singledispatch (Can use pip install for these).

Extracting an STL-model from pre-processed Learning and validate against Validation data. See thesis, Section 3.2.2.1 for in depth description.





How to use:
1. Datasets from pre-processing are placed in folders "learning_data" and "validation_data" respectively.

2. in learn.py:
2.1	Tune TeLEx parameters BETA, GAMMA
2.2	Choose STL-templates. 
	Either edit "run_for_variables" and "run_for_temporal_opperators"; 
	Or write custom STL-templates in "templogicdata". See TeLEx for syntax. Variable names must match columns in learning data. Custom templates may cause problems with the result printout.

3. Run learn.py with arguments
	> py -2.7 learn.py -i [itterations] -o "gradient"
	-i sets how many times to optimize each template.
	-o sets TeLEx optimization method. "gradient" is fastest.

The code is very slow! I get 10-15 min per template, per attempt. I suggest running multiple instances, for different variables.





Results are output to a new folder, "Result" or "Result_[x]" where [x] is the lowest number available. Results contain:
* about.log:	Notes on what parameters were used
* Result_gradient_[itterations].log:	Full table of results for each optimization attempt.
	Columns are:
	ID:	Not used
	Beta:	TeLEx parameter beta for optimization
	Gamma:	TeLEx parameter gamma for optimization
	LAvg Robustness:	Average robustness of Learning data signals against the STL-model (optimized template). Low, positive values indicate a tight fitting STL-model.
	LData fit (%):		Percentage of Learning data signals that satisfy the STL-model (optimized template). High percentage indicates a correct STL-model.
	VAvg Robustness:	Average robustness of Validation data signals against STL-model.
	VData fit (%):		Percentage of Validation data signals that satisfy the STL-model.
	Result:		STL model, containing optimized parameters.
* STL_[range].log:	List of STL model results, where "VData fit" falls in the range given in the file name.
	ex. 	STL_eq100.log contains STL-models satisfied by 100% of validation data.
		STL_90-95.log contains STL-models satisfied by greater than or equal to 90%, but less than 95% of validation data.
* STL_[range]_b.log:	Same list as above, formated as a string ready for the STL-based planner. (Will NOT work with custom STL-templates)
	


