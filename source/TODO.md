# Things to work on for AGN LF project

- Get K-corrections for the different bands included in the catalog
	- Probably can get these from Anna or Mark
- Implement new selection criteria for AGN
	- MIR: Lacy et al. 2005 and Lacy et al. 2015 (yr?)
		- Should be an okay starting point
	- Other bands: Idk
		- Look into selection criteria for other bands and how to do this synergistically
- Implement weight/selection functions to account for incompleteness of catalog
	- How is this usually done? => pull some papers
- Minimized $\chi^2$ method
	- Try scipy odr and simple distributions for initial parameters
- MCMC
	- Look at how to implement in emcee
		- See how Kulkarni go through the analysis

