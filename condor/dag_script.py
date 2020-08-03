class DagScript(object):
    template  = 'JOB {name} {submitScript}\n'
    #template += 'VARS {name} NAME="{name}"\n'
    #template += 'VARS {name} MEMORY="{memory}" DISK="{disc}" PRIORITY="{priority}"\n'
    template += 'VARS {name} {params}\n'
    #template += 'Retry {name} {retry}\n'
    #default_submitScript = "/path/to/some/shell/script.submit"
    attributes = ["name", "submitScript",  "params"]
    def __init__(self, **kwargs):
        """ This class generates a dagman file. Use the .add() function to add a job to the dagman file.
        Use the .write() function to save the file on disc.
        This function takes several optional parameters (default values for .add(...)):

            Optional Parameters:

            -------------------

                dagfile: Location where dagfile should be saved.
                submitScript: Location of the condor submit script.
                memory: requested memory in MB
                disc: requested disc space in MB
                priority: priotity of job execution (default: 1)
                retry: Number of retrys if a job fails
                executable: executable program location
                params: parameters passed to the script

        Note all parameters given here are default for add. 
        If these parameters are given in add as well the values given in add are used.
        """

        self.name = None
        self.dagfile = kwargs.pop("dagfile", None)
        self.submitScript = kwargs.pop("submitScript")#, DagScript.default_submitScript)
        #self.memory = kwargs.pop("memory", 1800)
        #self.disc = kwargs.pop("disc", 2000)
        #self.priority = kwargs.pop("priority", 1)
        #self.retry = kwargs.pop("retry", 1)
        #self.executable =  kwargs.pop("executable", None)
        self.params = kwargs.pop("params", None)
        self.list_of_jobs = []

    def add(self, **kwargs):
        """ 
        Add a job to the dagfile:

        Optional Parameters:
        -------------------
        submitScript: Location of the condor submit script.
        memory: requested memory in MB
        disc: requested disc space in MB
        priority: priotity of job execution (default: 1)
        retry: Number of retrys if a job fails
        executable: executable program location
        params: parameters passed to the script

        If not given default parameters are used. 

        Raises:
        -----
        RuntimeWarning: If any required attribute is not given by default or as argument 
        a RuntimeWarning is raised and the add is skipped.
        """

        entry = {}
        for attr in DagScript.attributes:
            entry[attr] = kwargs.pop(attr, getattr(self, attr))
            if entry[attr] is None:
                raise RuntimeWarning("The following value '%s' is not defined. We have to skip this entry."%attr)
        self.list_of_jobs.append(entry)

    def write(self, dagfile=None):
        """ 
        Save the dagfile to disc. 
        Optional Parameters:
        --------------------            
        dagfile: Location where dagfile should be saved.
        Note if dagfile is not given the default from __init__ call will be used.
        Raises:
        -------
        RuntimeWarning: If dagfile is not given by default or as argument a 
        RuntimeWarning is raised and no file is written to disc.
        """

        if dagfile is None and self.dagfile is None:
            raise RuntimeWarning("No dagfile was specified. Please specify a dagfile. Skipped.")
        with open(dagfile, "w") as open_file:
            for entry in self.list_of_jobs:
                name = entry["name"]
                submitScript = entry["submitScript"]
                #memory = entry["memory"]
                #disc = entry["disc"]
                #priority = entry["priority"]
                #executable = entry["executable"]
                params = entry["params"]
                #retry = entry["retry"]
                open_file.write(DagScript.template.format(**locals()))