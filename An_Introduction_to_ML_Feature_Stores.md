# An Introduction to ML Feature Stores

<em>This is the first article in a series of articles on ML feature stores. The
series will detail what a feature store is, the challenges a feature store must
address, the components of a feature store, how those components address feature
store challenges, and solutions to some of the common technical problems faced
by feature stores. These articles are intended for software engineers, but any
technical professional with an interest in software engineering or in a software
engineering adjacent profession should find them approachable. If you don’t know
anything about feature stores, by the conclusion of this series you should have
a solid understanding of what feature stores do, why they exist, and how they
address some of their challenges. If you are already familiar with feature
stores, then this series may introduce you to some new aspects of feature
stores.</em>

---

## Introduction

**What is a feature store?** This is the question that will be fully answered by
this article, but briefly a feature store is a new piece of infrastructure in
the burgeoning field of Machine Learning Operations — or MLOps. The components
of a feature store are responsible for managing the lifecycle of features
utilized in machine learning inference. In the same way that DevOps and tools
like Terraform have brought infrastructure configuration management capabilities
to infrastructure configuration, MLOps is bringing feature lifecycle management
capabilities to machine learning.

In order for the above description to make any sense, some of the terms used in
the above description should be defined for those less familiar with the space.

<u>Machine Learning Operations (MLOps):</u> MLOps is concerned with the
inclusion of machine learning capabilities in production software applications.
Like other components of a production software application, these machine
learning capabilities must be testable, upgradeable, debuggable, and have
predictable performance characteristics.

<u>Feature:</u> A feature is a list of feature values. Each feature value is a
property of an identifiable data entity. For example, for a customer data
entity, the age of the customer could be a feature value; for a product data
entity, the cost of the product could be a feature value. Feature values can
also be aggregate values. For example, an aggregate feature value could be the
total value of all purchases made by a customer within a month, computed for
every month for which there is data; or the total number of times a product was
sold within a specific city, computed for every product and city for which there
is data. Finally, features can be calculated from other features. Generally, a
feature value needs to be coupled with a timestamp, indicating when the feature
value was first known.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*v_sIouHfeIdkp2bQXEBEdw.png)

Informally, if the rows in a relational database table correspond to
identifiable data entities, the value of a specific row and column is a feature
value, and the values in a column for all rows is a feature.

**Why do we need a feature store?** Given the rapid advances being made in
machine learning and the availability of flexible compute provided by cloud
providers, enterprises of all sizes are integrating machine learning
capabilities into their software products. As organizations integrate machine
learning capabilities into products, a common set of challenges has emerged.
These common challenges have given rise to a new category of infrastructure
targeted at addressing the challenges of operational machine learning — the
feature store.

### Online/Offline

A feature store can be used in online and/or offline scenarios.

In an **online** scenario a feature store is used to provide features to a
system performing machine learning inference in response to client requests.

In an **offline** scenario a feature store is used to provide features to a
system performing machine learning inference in batch on persisted data.

Online systems are sensitive to latency, as quickly producing inference results
in response to many simultaneous client requests is paramount. Offline systems
focus on volume, the latency of one inference result is not as important as
being able to process a large volume of inference results in a reasonable amount
of time. If processing time requirements are satisfied, offline inference
results can be produced sequentially.

### The Feature Lifecycle

Generally the feature lifecycle consists of the following stages:

- Feature definition
- Feature generation
- Feature evaluation
- Feature use
- Feature inactivation

The feature lifecycle is not a strictly linear lifecycle. Instead, a feature may
rapidly cycle between stages or return to a previous stage at regular intervals
or in response to external changes. For example, early in its existence a
feature may rapidly cycle through the feature definition, generation, and
evaluation stages as the feature is defined, generated, evaluated, and then
redefined as a result of evaluation. Later, after a feature has been deployed
for production use, its accuracy may be re-evaluated at regular intervals,
possibly resulting in it re-entering the feature definition, generation, and
evaluation cycle.

![The Feature Lifecycle](https://miro.medium.com/v2/resize:fit:720/format:webp/1*U9PFLzdN32998E8tRZ0_dg.png)

**Feature Definition** During feature definition the source data and
calculations that produce a feature are defined. All of the aspects necessary to
generate and/or re-generate a feature must be captured during this stage, as
these definitions will be used during all subsequent feature lifecycle stages.
The feature definition includes a feature version or new feature version when a
feature’s definition has changed. Additionally, it is during feature definition
when any security controls limiting feature access are defined.

**Feature Generation** During feature generation the definition of a feature is
executed to produce the feature. Ideally, the feature generation stage should
support complete, incremental, and partial feature generation.

**_Complete_** feature generation or regeneration results in replacement of all
existing feature values for a feature. Complete feature generation takes place
when a feature is newly defined or the definition of a feature has changed.

**_Incremental_** feature generation results in replacement of only a portion of
feature values or the inclusion of new feature values alongside existing feature
values. Incremental feature generation takes place when new source data becomes
available. For example, incremental feature generation can be used to generate
feature values for new timestamps that were in the future (or past) and
unavailable when the feature was originally defined and generated. Incremental
feature generation allows new feature values to be quickly and easily generated.

**_Partial_** feature generation results in incomplete calculation of a feature
value with complete calculation pending availability of additional source data
values. Frequently, these additional source data values are provided with a
client request.

**Feature Evaluation** During feature evaluation, a feature (or features) are
retrieved from the feature store and used to train a machine learning model
and/or evaluate the accuracy of a machine learning model. Retrieval of features
during evaluation is subject to any feature access controls defined during the
feature definition stage.

**Feature Use** During feature use, a feature is repeatedly retrieved from a
feature store and provided to a machine learning model. As with the feature
evaluation stage, any feature security controls are enforced during feature use.

**Feature Inactivation** Feature inactivation is the end of the feature
lifecycle. Feature inactivation allows a feature version or all in-use feature
versions (there may be multiple version of a feature in use at the same time) to
be deprecated or inactivated. Marking a feature for deprecation prevents a
feature from being used in new feature definitions. Marking a feature for
inactivation prevents a feature from being retrieved.

## Feature Store Challenges

ML feature stores are concerned with managing the feature lifecycle, providing
features to online and offline systems, and recording feature utilization and
performance monitoring statistics.

A feature store isn’t necessary when

- There are only a few features (tens);

- There are only a few values for each feature (thousands);

- Features can be quickly and easily calculated;

- Features are consumed by only one or a few applications; and/or

- There is only one version of a feature or it’s not necessary to manage
  multiple versions of a feature.

However, a feature store should be considered

- There are many features (thousands);

- There are many features for each value (tens of thousands to millions);

- Features cannot be easily recalculated;

- Features are consumed by many applications;

- Features are shared by multiple teams across the organization; and/or

- There may be multiple versions of a feature in use simultaneously.

The challenges faced by a feature store can be broadly grouped into two
categories — _technical_ and _operational_.

### Technical Challenges

Technical challenges are those challenges concerned with including machine
learning inference as part of the functionality of a modern
application—operational machine learning or MLOps. These challenges are
concerned with the calculation, evaluation, and use of features by online and
offline systems.

**_Online_** systems are concerned with generating inference results with
minimal latency. As far as a feature store is concerned, this means enabling low
latency retrieval of features at scale. Additionally, feature stores need to be
able to perform last-mile feature calculations by combining pre-computed,
intermediate feature data with data that is available only at client request
time. For example, a feature used to flag fraudulent transactions may use the
category of a purchased product. The set of product categories can be
pre-computed, but assignment of a purchase to a product category cannot take
place until the product is communicated in a client request. Strategies for the
low latency calculation of features from intermediate feature data will be
discussed in a later article focusing on that subject.

**_Offline_** systems are primarily concerned with the ability to rapidly
iterate through the feature calculation and model evaluation cycle. Solutions to
these challenges must ensure consistency with online solutions. For example, at
training-time offline solutions must be aware of the data available to an online
system at a specific point-in-time so as not to bias model training by exposing
the model to data that would not be available for online inference. Despite
offline challenges being concerned primarily with accelerating model
development, offline systems may also be used for batch inference; therefore,
offline systems are subject to some of the same challenges as online systems.

### Operational Challenges

Operational challenges are those challenges concerned with enabling the use of
features throughout the enterprise. The challenges include cataloging features,
enforcing feature access controls, monitoring feature performance, and recording
feature utilization.

A comprehensive, centralized, and searchable catalog of features is necessary to
enable the use of features throughout the enterprise. Features are cataloged by
their metadata. Feature metadata can include feature descriptions, versions,
ownership information, access restrictions, generation timestamps, update
frequency, provenance, intended use cases, etc. A rich set of feature metadata
enables sophisticated feature search capabilities and allows feature consumers
to accurately evaluate features. The feature catalog can enable an enterprises
to achieve economies of scale in their feature use by decoupling feature
producers from feature consumers.

Features may contain highly sensitive or valuable information; therefore, it is
critical that it be possible to define and enforce access restrictions to
features and feature metadata.

Feature performance monitoring allows an enterprise to establish feature
SLAs—the frequency with which new feature data is generated from new source
data, the latency with which features are provided to online or offline systems,
etc.—and audit adherence to these SLAs. Additionally, performance and
utilization data allows an enterprise make informed tradeoffs when it comes to
the cost of supporting alternative features or feature versions.

---

<em>In the next article in this series the feature lifecycle stages will be
mapped onto software components that together manage the feature lifecycle and
satisfy the technical and operational challenges of a feature store.</em>
