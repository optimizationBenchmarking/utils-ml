# License of the Optimization Benchmarking Library `utils-ml`

The `utils-ml` library (from now on referred to as *the library*) of the *Optimization Benchmarking Project* is licensed under the [GNU General Public License Version 3](http://www.gnu.org/meta/licenses/gpl-3.0-standalone.html), 29 June 2007. This license applies to all of the software and all other parts of *the library*, except for the exceptions listed under point "Exceptions" below.

The text of the GNU General Public License Version 3 is provided in the file "[meta/licenses/gpl-3.0.txt](http://github.com/optimizationBenchmarking/utils-ml/tree/master/meta/licenses/gpl-3.0.txt)" and can be found online at [http://www.gnu.org/meta/licenses/gpl-3.0.txt](http://www.gnu.org/meta/licenses/gpl-3.0.txt).

## Exceptions

### Referenced Libraries

*The library* _directly_ depends on a set of external libraries. These will automatically be loaded if you install the project via the [Maven POM](http://github.com/optimizationBenchmarking/utils-graphics/tree/master/pom.xml) file. They are not part of the official source code. However, they may be included in "fat `jars`", i.e., the `jar` archives we create via the [Maven Shade Plugin](http://maven.apache.org/plugins/maven-shade-plugin/) and which hold a "-full" in their name. These `jar` archive are created for the user's convenience: Using them, no additional libraries need to be put into the classpath.

The following libraries are needed by *the library*:


* `commons-math3-3.5.jar`
    - License: [The Apache Software License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.txt) (see file "[meta/licenses/apache-2.0.txt](http://github.com/optimizationBenchmarking/utils-ml/tree/master/meta/licenses/apache-2.0.txt)")
    - Version: 3.5
    - Status: freely available in the internet
    - Source: [repo.maven.apache.org](http://repo.maven.apache.org/maven2/org/apache/commons/commons-math3/3.5/commons-math3-3.5.jar)
    
* `hamcrest-core-1.3.jar`
    - License: [New BSD License](http://www.opensource.org/licenses/bsd-license.php) (see file "[meta/licenses/bsd-license.txt](http://github.com/optimizationBenchmarking/utils-ml/tree/master/meta/licenses/bsd-license.txt)")
    - Version: 1.3
    - Status: freely available in the internet
    - Source: [repo.maven.apache.org](http://repo.maven.apache.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar)

* `junit-4.11.jar`
    - License: [Common Public License Version 1.0](http://www.opensource.org/licenses/cpl1.0.txt) (see file "[meta/licenses/cpl-1.0.txt](http://github.com/optimizationBenchmarking/utils-ml/tree/master/meta/licenses/cpl-1.0.txt)")
    - Version: 4.11
    - Status: freely available in the internet
    - Source: [repo.maven.apache.org](http://repo.maven.apache.org/maven2/junit/junit/4.11/junit-4.11.jar)

## Questions and Problems

If you have any questions about *the library*, want to contribute to *the library*, want to discuss licensing terms, or have any problem with *the project* itself and/or its licensing terms, please contact the project main author via [email](mailto:tweise@ustc.edu.cn) (see point "Contact").

## Contact

The main author, copyright holder, and corresponding author of *the library* is Dr. Thomas Weise.

**Dr. Thomas Weise**
Nature Inspired Computation and Applications Laboratory (NICAL)
USTC-Birmingham Joint Research Institute in Intelligent Computation and Its Applications (UBRI)
School of Computer Science and Technology (SCST)
University of Science and Technology of China (USTC)
West Campus, Science and Technology Building, West Wing, Room 601
Huangshan Road/Feixi Road, Hefei 230027, Anhui, China
Web:    [http://www.it-weise.de/](http://www.it-weise.de/)
Email:  [tweise@gmx.de](mailto:tweise@gmx.de), [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn)
