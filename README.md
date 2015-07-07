Topic Classifier
================

Uses LSA for topic extraction and SVM for classification.

1. Launch sbt:

        cd scala-sbt-template
        ./sbt (or sbt.bat for Windows)
        
  This downloads all the dependencies for the project.

2. To run your program, in SBT:
   
        >run        
        
3. To load the project into Eclipse, at the SBT prompt:

        > eclipse with-sources=true
        
   Then from within Eclipse, select File->Import->General->Existing Projects Into Workspace, and select your project directory. 
   Install the Eclipse Scala IDE plugin from [here](http://scala-ide.org/download/current.html).

4. Or, to load the project into Intellij IDEA, at the SBT prompt:
   
        > gen-idea
        
   Then from within IDEA, select File -> Open Project, and select your project directory.

5. To run all your tests:

        > test
        
