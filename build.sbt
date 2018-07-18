lazy val sparkVersion = settingKey[String]("The version of Spark used for the project")
lazy val dl4jVersion = settingKey[String]("The version of DL4J used for the project")

lazy val rl4jpractice =
  (project in file(".")).
    settings(
      name := "RL4JPractice",
      organization := "net.ddns.akgunter",
      version := "0.1",
      scalaVersion := "2.11.12",
      sparkVersion := "2.3.1",
      dl4jVersion := "1.0.0-beta",
      test in assembly := {},
      assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false),
      assemblyMergeStrategy in assembly := {
        case PathList("log4j.properties") => MergeStrategy.discard
        case PathList("log4j.xml") => MergeStrategy.discard
        case x =>
          val oldStrategy = (assemblyMergeStrategy in assembly).value
          oldStrategy(x)
      },
      libraryDependencies ++= Seq(
        "org.apache.spark" %% "spark-sql" % sparkVersion.value % "provided",
        "org.nd4j" % "nd4j-native-platform" % dl4jVersion.value,
        "org.deeplearning4j" % "deeplearning4j-parent" % dl4jVersion.value,
        "org.deeplearning4j" % "rl4j" % dl4jVersion.value,
        "org.datavec" % "datavec-parent" % dl4jVersion.value
      )
    )