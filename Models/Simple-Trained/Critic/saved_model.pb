Ľ
ÝŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.0-dev202007112v1.12.1-36285-g51be86b23b8ĐÎ

critic_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_namecritic_model/dense_2/kernel

/critic_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpcritic_model/dense_2/kernel*
_output_shapes

:2*
dtype0

critic_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2**
shared_namecritic_model/dense_2/bias

-critic_model/dense_2/bias/Read/ReadVariableOpReadVariableOpcritic_model/dense_2/bias*
_output_shapes
:2*
dtype0

critic_model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_namecritic_model/dense_3/kernel

/critic_model/dense_3/kernel/Read/ReadVariableOpReadVariableOpcritic_model/dense_3/kernel*
_output_shapes

:2*
dtype0

critic_model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecritic_model/dense_3/bias

-critic_model/dense_3/bias/Read/ReadVariableOpReadVariableOpcritic_model/dense_3/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
˘
#Nadam/critic_model/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#Nadam/critic_model/dense_2/kernel/m

7Nadam/critic_model/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp#Nadam/critic_model/dense_2/kernel/m*
_output_shapes

:2*
dtype0

!Nadam/critic_model/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!Nadam/critic_model/dense_2/bias/m

5Nadam/critic_model/dense_2/bias/m/Read/ReadVariableOpReadVariableOp!Nadam/critic_model/dense_2/bias/m*
_output_shapes
:2*
dtype0
˘
#Nadam/critic_model/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#Nadam/critic_model/dense_3/kernel/m

7Nadam/critic_model/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp#Nadam/critic_model/dense_3/kernel/m*
_output_shapes

:2*
dtype0

!Nadam/critic_model/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/critic_model/dense_3/bias/m

5Nadam/critic_model/dense_3/bias/m/Read/ReadVariableOpReadVariableOp!Nadam/critic_model/dense_3/bias/m*
_output_shapes
:*
dtype0
˘
#Nadam/critic_model/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#Nadam/critic_model/dense_2/kernel/v

7Nadam/critic_model/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp#Nadam/critic_model/dense_2/kernel/v*
_output_shapes

:2*
dtype0

!Nadam/critic_model/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!Nadam/critic_model/dense_2/bias/v

5Nadam/critic_model/dense_2/bias/v/Read/ReadVariableOpReadVariableOp!Nadam/critic_model/dense_2/bias/v*
_output_shapes
:2*
dtype0
˘
#Nadam/critic_model/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#Nadam/critic_model/dense_3/kernel/v

7Nadam/critic_model/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp#Nadam/critic_model/dense_3/kernel/v*
_output_shapes

:2*
dtype0

!Nadam/critic_model/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/critic_model/dense_3/bias/v

5Nadam/critic_model/dense_3/bias/v/Read/ReadVariableOpReadVariableOp!Nadam/critic_model/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ő
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*°
valueŚBŁ B


dense1
	value
opt
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

iter

beta_1

beta_2
	decay
learning_rate
momentum_cache	m*
m+m,m-	v.
v/v0v1

	0

1
2
3
 

	0

1
2
3
­
	variables
non_trainable_variables

layers
metrics
regularization_losses
layer_metrics
layer_regularization_losses
trainable_variables
 
YW
VARIABLE_VALUEcritic_model/dense_2/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcritic_model/dense_2/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
­
	variables
 non_trainable_variables

!layers
"metrics
regularization_losses
#layer_metrics
$layer_regularization_losses
trainable_variables
XV
VARIABLE_VALUEcritic_model/dense_3/kernel'value/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcritic_model/dense_3/bias%value/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
%non_trainable_variables

&layers
'metrics
regularization_losses
(layer_metrics
)layer_regularization_losses
trainable_variables
CA
VARIABLE_VALUE
Nadam/iter#opt/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUENadam/beta_1%opt/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUENadam/beta_2%opt/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUENadam/decay$opt/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUENadam/learning_rate,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUENadam/momentum_cache-opt/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUE#Nadam/critic_model/dense_2/kernel/m>dense1/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!Nadam/critic_model/dense_2/bias/m<dense1/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE#Nadam/critic_model/dense_3/kernel/m=value/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!Nadam/critic_model/dense_3/bias/m;value/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE#Nadam/critic_model/dense_2/kernel/v>dense1/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!Nadam/critic_model/dense_2/bias/v<dense1/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE#Nadam/critic_model/dense_3/kernel/v=value/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!Nadam/critic_model/dense_3/bias/v;value/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Ż
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1critic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3073574
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/critic_model/dense_2/kernel/Read/ReadVariableOp-critic_model/dense_2/bias/Read/ReadVariableOp/critic_model/dense_3/kernel/Read/ReadVariableOp-critic_model/dense_3/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp7Nadam/critic_model/dense_2/kernel/m/Read/ReadVariableOp5Nadam/critic_model/dense_2/bias/m/Read/ReadVariableOp7Nadam/critic_model/dense_3/kernel/m/Read/ReadVariableOp5Nadam/critic_model/dense_3/bias/m/Read/ReadVariableOp7Nadam/critic_model/dense_2/kernel/v/Read/ReadVariableOp5Nadam/critic_model/dense_2/bias/v/Read/ReadVariableOp7Nadam/critic_model/dense_3/kernel/v/Read/ReadVariableOp5Nadam/critic_model/dense_3/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3073730

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cache#Nadam/critic_model/dense_2/kernel/m!Nadam/critic_model/dense_2/bias/m#Nadam/critic_model/dense_3/kernel/m!Nadam/critic_model/dense_3/bias/m#Nadam/critic_model/dense_2/kernel/v!Nadam/critic_model/dense_2/bias/v#Nadam/critic_model/dense_3/kernel/v!Nadam/critic_model/dense_3/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3073794Šű
ÁW
Ž
"__inference__wrapped_model_3073445
input_1:
6critic_model_dense_2_tensordot_readvariableop_resource8
4critic_model_dense_2_biasadd_readvariableop_resource:
6critic_model_dense_3_tensordot_readvariableop_resource8
4critic_model_dense_3_biasadd_readvariableop_resource
identity|
critic_model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
critic_model/ExpandDims/dimĽ
critic_model/ExpandDims
ExpandDimsinput_1$critic_model/ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
critic_model/ExpandDimsŐ
-critic_model/dense_2/Tensordot/ReadVariableOpReadVariableOp6critic_model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02/
-critic_model/dense_2/Tensordot/ReadVariableOp
#critic_model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_model/dense_2/Tensordot/axes
#critic_model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#critic_model/dense_2/Tensordot/free
$critic_model/dense_2/Tensordot/ShapeShape critic_model/ExpandDims:output:0*
T0*
_output_shapes
:2&
$critic_model/dense_2/Tensordot/Shape
,critic_model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_2/Tensordot/GatherV2/axisş
'critic_model/dense_2/Tensordot/GatherV2GatherV2-critic_model/dense_2/Tensordot/Shape:output:0,critic_model/dense_2/Tensordot/free:output:05critic_model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/GatherV2˘
.critic_model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_2/Tensordot/GatherV2_1/axisŔ
)critic_model/dense_2/Tensordot/GatherV2_1GatherV2-critic_model/dense_2/Tensordot/Shape:output:0,critic_model/dense_2/Tensordot/axes:output:07critic_model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)critic_model/dense_2/Tensordot/GatherV2_1
$critic_model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$critic_model/dense_2/Tensordot/ConstÔ
#critic_model/dense_2/Tensordot/ProdProd0critic_model/dense_2/Tensordot/GatherV2:output:0-critic_model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#critic_model/dense_2/Tensordot/Prod
&critic_model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&critic_model/dense_2/Tensordot/Const_1Ü
%critic_model/dense_2/Tensordot/Prod_1Prod2critic_model/dense_2/Tensordot/GatherV2_1:output:0/critic_model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%critic_model/dense_2/Tensordot/Prod_1
*critic_model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*critic_model/dense_2/Tensordot/concat/axis
%critic_model/dense_2/Tensordot/concatConcatV2,critic_model/dense_2/Tensordot/free:output:0,critic_model/dense_2/Tensordot/axes:output:03critic_model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%critic_model/dense_2/Tensordot/concatŕ
$critic_model/dense_2/Tensordot/stackPack,critic_model/dense_2/Tensordot/Prod:output:0.critic_model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_2/Tensordot/stacké
(critic_model/dense_2/Tensordot/transpose	Transpose critic_model/ExpandDims:output:0.critic_model/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(critic_model/dense_2/Tensordot/transposeó
&critic_model/dense_2/Tensordot/ReshapeReshape,critic_model/dense_2/Tensordot/transpose:y:0-critic_model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2(
&critic_model/dense_2/Tensordot/Reshapeň
%critic_model/dense_2/Tensordot/MatMulMatMul/critic_model/dense_2/Tensordot/Reshape:output:05critic_model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙22'
%critic_model/dense_2/Tensordot/MatMul
&critic_model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22(
&critic_model/dense_2/Tensordot/Const_2
,critic_model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_2/Tensordot/concat_1/axisŚ
'critic_model/dense_2/Tensordot/concat_1ConcatV20critic_model/dense_2/Tensordot/GatherV2:output:0/critic_model/dense_2/Tensordot/Const_2:output:05critic_model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/concat_1ä
critic_model/dense_2/TensordotReshape/critic_model/dense_2/Tensordot/MatMul:product:00critic_model/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22 
critic_model/dense_2/TensordotË
+critic_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+critic_model/dense_2/BiasAdd/ReadVariableOpŰ
critic_model/dense_2/BiasAddBiasAdd'critic_model/dense_2/Tensordot:output:03critic_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
critic_model/dense_2/BiasAdd
critic_model/dense_2/ReluRelu%critic_model/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
critic_model/dense_2/ReluŐ
-critic_model/dense_3/Tensordot/ReadVariableOpReadVariableOp6critic_model_dense_3_tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02/
-critic_model/dense_3/Tensordot/ReadVariableOp
#critic_model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_model/dense_3/Tensordot/axes
#critic_model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#critic_model/dense_3/Tensordot/freeŁ
$critic_model/dense_3/Tensordot/ShapeShape'critic_model/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$critic_model/dense_3/Tensordot/Shape
,critic_model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_3/Tensordot/GatherV2/axisş
'critic_model/dense_3/Tensordot/GatherV2GatherV2-critic_model/dense_3/Tensordot/Shape:output:0,critic_model/dense_3/Tensordot/free:output:05critic_model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/GatherV2˘
.critic_model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_3/Tensordot/GatherV2_1/axisŔ
)critic_model/dense_3/Tensordot/GatherV2_1GatherV2-critic_model/dense_3/Tensordot/Shape:output:0,critic_model/dense_3/Tensordot/axes:output:07critic_model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)critic_model/dense_3/Tensordot/GatherV2_1
$critic_model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$critic_model/dense_3/Tensordot/ConstÔ
#critic_model/dense_3/Tensordot/ProdProd0critic_model/dense_3/Tensordot/GatherV2:output:0-critic_model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#critic_model/dense_3/Tensordot/Prod
&critic_model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&critic_model/dense_3/Tensordot/Const_1Ü
%critic_model/dense_3/Tensordot/Prod_1Prod2critic_model/dense_3/Tensordot/GatherV2_1:output:0/critic_model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%critic_model/dense_3/Tensordot/Prod_1
*critic_model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*critic_model/dense_3/Tensordot/concat/axis
%critic_model/dense_3/Tensordot/concatConcatV2,critic_model/dense_3/Tensordot/free:output:0,critic_model/dense_3/Tensordot/axes:output:03critic_model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%critic_model/dense_3/Tensordot/concatŕ
$critic_model/dense_3/Tensordot/stackPack,critic_model/dense_3/Tensordot/Prod:output:0.critic_model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_3/Tensordot/stackđ
(critic_model/dense_3/Tensordot/transpose	Transpose'critic_model/dense_2/Relu:activations:0.critic_model/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
(critic_model/dense_3/Tensordot/transposeó
&critic_model/dense_3/Tensordot/ReshapeReshape,critic_model/dense_3/Tensordot/transpose:y:0-critic_model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2(
&critic_model/dense_3/Tensordot/Reshapeň
%critic_model/dense_3/Tensordot/MatMulMatMul/critic_model/dense_3/Tensordot/Reshape:output:05critic_model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%critic_model/dense_3/Tensordot/MatMul
&critic_model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&critic_model/dense_3/Tensordot/Const_2
,critic_model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_3/Tensordot/concat_1/axisŚ
'critic_model/dense_3/Tensordot/concat_1ConcatV20critic_model/dense_3/Tensordot/GatherV2:output:0/critic_model/dense_3/Tensordot/Const_2:output:05critic_model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/concat_1ä
critic_model/dense_3/TensordotReshape/critic_model/dense_3/Tensordot/MatMul:product:00critic_model/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
critic_model/dense_3/TensordotË
+critic_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+critic_model/dense_3/BiasAdd/ReadVariableOpŰ
critic_model/dense_3/BiasAddBiasAdd'critic_model/dense_3/Tensordot:output:03critic_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
critic_model/dense_3/BiasAdd}
IdentityIdentity%critic_model/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ě
~
)__inference_dense_3_layer_call_fn_3073653

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_30735282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
 
_user_specified_nameinputs


%__inference_signature_wrapper_3073574
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_30734452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
0
	
 __inference__traced_save_3073730
file_prefix:
6savev2_critic_model_dense_2_kernel_read_readvariableop8
4savev2_critic_model_dense_2_bias_read_readvariableop:
6savev2_critic_model_dense_3_kernel_read_readvariableop8
4savev2_critic_model_dense_3_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableopB
>savev2_nadam_critic_model_dense_2_kernel_m_read_readvariableop@
<savev2_nadam_critic_model_dense_2_bias_m_read_readvariableopB
>savev2_nadam_critic_model_dense_3_kernel_m_read_readvariableop@
<savev2_nadam_critic_model_dense_3_bias_m_read_readvariableopB
>savev2_nadam_critic_model_dense_2_kernel_v_read_readvariableop@
<savev2_nadam_critic_model_dense_2_bias_v_read_readvariableopB
>savev2_nadam_critic_model_dense_3_kernel_v_read_readvariableop@
<savev2_nadam_critic_model_dense_3_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1edda3004ac74c91aca26aabaa870250/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename­
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ż
valueľB˛B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB%opt/beta_1/.ATTRIBUTES/VARIABLE_VALUEB%opt/beta_2/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-opt/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB>dense1/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB<dense1/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB=value/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB;value/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB>dense1/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB<dense1/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB=value/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB;value/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesŽ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices˘	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_critic_model_dense_2_kernel_read_readvariableop4savev2_critic_model_dense_2_bias_read_readvariableop6savev2_critic_model_dense_3_kernel_read_readvariableop4savev2_critic_model_dense_3_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop>savev2_nadam_critic_model_dense_2_kernel_m_read_readvariableop<savev2_nadam_critic_model_dense_2_bias_m_read_readvariableop>savev2_nadam_critic_model_dense_3_kernel_m_read_readvariableop<savev2_nadam_critic_model_dense_3_bias_m_read_readvariableop>savev2_nadam_critic_model_dense_2_kernel_v_read_readvariableop<savev2_nadam_critic_model_dense_2_bias_v_read_readvariableop>savev2_nadam_critic_model_dense_3_kernel_v_read_readvariableop<savev2_nadam_critic_model_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesr
p: :2:2:2:: : : : : : :2:2:2::2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
˛
˘
.__inference_critic_model_layer_call_fn_3073559
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_critic_model_layer_call_and_return_conditional_losses_30735452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ŕ
Ż
D__inference_dense_2_layer_call_and_return_conditional_losses_3073482

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis˝
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙:::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´O
Ţ

#__inference__traced_restore_3073794
file_prefix0
,assignvariableop_critic_model_dense_2_kernel0
,assignvariableop_1_critic_model_dense_2_bias2
.assignvariableop_2_critic_model_dense_3_kernel0
,assignvariableop_3_critic_model_dense_3_bias!
assignvariableop_4_nadam_iter#
assignvariableop_5_nadam_beta_1#
assignvariableop_6_nadam_beta_2"
assignvariableop_7_nadam_decay*
&assignvariableop_8_nadam_learning_rate+
'assignvariableop_9_nadam_momentum_cache;
7assignvariableop_10_nadam_critic_model_dense_2_kernel_m9
5assignvariableop_11_nadam_critic_model_dense_2_bias_m;
7assignvariableop_12_nadam_critic_model_dense_3_kernel_m9
5assignvariableop_13_nadam_critic_model_dense_3_bias_m;
7assignvariableop_14_nadam_critic_model_dense_2_kernel_v9
5assignvariableop_15_nadam_critic_model_dense_2_bias_v;
7assignvariableop_16_nadam_critic_model_dense_3_kernel_v9
5assignvariableop_17_nadam_critic_model_dense_3_bias_v
identity_19˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ł
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ż
valueľB˛B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB%opt/beta_1/.ATTRIBUTES/VARIABLE_VALUEB%opt/beta_2/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-opt/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB>dense1/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB<dense1/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB=value/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB;value/bias/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUEB>dense1/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB<dense1/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB=value/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB;value/bias/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityŤ
AssignVariableOpAssignVariableOp,assignvariableop_critic_model_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ą
AssignVariableOp_1AssignVariableOp,assignvariableop_1_critic_model_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ł
AssignVariableOp_2AssignVariableOp.assignvariableop_2_critic_model_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ą
AssignVariableOp_3AssignVariableOp,assignvariableop_3_critic_model_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4˘
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ł
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ť
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ź
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ż
AssignVariableOp_10AssignVariableOp7assignvariableop_10_nadam_critic_model_dense_2_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11˝
AssignVariableOp_11AssignVariableOp5assignvariableop_11_nadam_critic_model_dense_2_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ż
AssignVariableOp_12AssignVariableOp7assignvariableop_12_nadam_critic_model_dense_3_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13˝
AssignVariableOp_13AssignVariableOp5assignvariableop_13_nadam_critic_model_dense_3_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ż
AssignVariableOp_14AssignVariableOp7assignvariableop_14_nadam_critic_model_dense_2_kernel_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15˝
AssignVariableOp_15AssignVariableOp5assignvariableop_15_nadam_critic_model_dense_2_bias_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ż
AssignVariableOp_16AssignVariableOp7assignvariableop_16_nadam_critic_model_dense_3_kernel_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17˝
AssignVariableOp_17AssignVariableOp5assignvariableop_17_nadam_critic_model_dense_3_bias_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpę
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18Ý
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Ż
D__inference_dense_3_layer_call_and_return_conditional_losses_3073528

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis˝
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2:::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
 
_user_specified_nameinputs
Á

I__inference_critic_model_layer_call_and_return_conditional_losses_3073545
input_1
dense_2_3073493
dense_2_3073495
dense_3_3073539
dense_3_3073541
identity˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

ExpandDimsŁ
dense_2/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0dense_2_3073493dense_2_3073495*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_30734822!
dense_2/StatefulPartitionedCall¸
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3073539dense_3_3073541*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_30735282!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ŕ
Ż
D__inference_dense_2_layer_call_and_return_conditional_losses_3073605

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis˝
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙:::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě
~
)__inference_dense_2_layer_call_fn_3073614

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_30734822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ż
D__inference_dense_3_layer_call_and_return_conditional_losses_3073644

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis˝
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2:::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
 
_user_specified_nameinputs"ĘL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ż
serving_default
;
input_10
serving_default_input_1:0˙˙˙˙˙˙˙˙˙@
output_14
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:B
ŕ

dense1
	value
opt
	variables
regularization_losses
trainable_variables
	keras_api

signatures
2__call__
3_default_save_signature
*4&call_and_return_all_conditional_losses"
_tf_keras_modelę{"class_name": "CriticModel", "name": "critic_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CriticModel"}}
ď

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"Ę
_tf_keras_layer°{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 1, 1]}}
ň

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"Í
_tf_keras_layerł{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 1, 50]}}
Ż
iter

beta_1

beta_2
	decay
learning_rate
momentum_cache	m*
m+m,m-	v.
v/v0v1"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
Ę
	variables
non_trainable_variables

layers
metrics
regularization_losses
layer_metrics
layer_regularization_losses
trainable_variables
2__call__
3_default_save_signature
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
-:+22critic_model/dense_2/kernel
':%22critic_model/dense_2/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
	variables
 non_trainable_variables

!layers
"metrics
regularization_losses
#layer_metrics
$layer_regularization_losses
trainable_variables
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
-:+22critic_model/dense_3/kernel
':%2critic_model/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
%non_trainable_variables

&layers
'metrics
regularization_losses
(layer_metrics
)layer_regularization_losses
trainable_variables
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
3:122#Nadam/critic_model/dense_2/kernel/m
-:+22!Nadam/critic_model/dense_2/bias/m
3:122#Nadam/critic_model/dense_3/kernel/m
-:+2!Nadam/critic_model/dense_3/bias/m
3:122#Nadam/critic_model/dense_2/kernel/v
-:+22!Nadam/critic_model/dense_2/bias/v
3:122#Nadam/critic_model/dense_3/kernel/v
-:+2!Nadam/critic_model/dense_3/bias/v
ü2ů
.__inference_critic_model_layer_call_fn_3073559Ć
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
ŕ2Ý
"__inference__wrapped_model_3073445ś
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
2
I__inference_critic_model_layer_call_and_return_conditional_losses_3073545Ć
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ó2Đ
)__inference_dense_2_layer_call_fn_3073614˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_3073605˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ó2Đ
)__inference_dense_3_layer_call_fn_3073653˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_3_layer_call_and_return_conditional_losses_3073644˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
4B2
%__inference_signature_wrapper_3073574input_1
"__inference__wrapped_model_3073445q	
0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş "7Ş4
2
output_1&#
output_1˙˙˙˙˙˙˙˙˙°
I__inference_critic_model_layer_call_and_return_conditional_losses_3073545c	
0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş ")˘&

0˙˙˙˙˙˙˙˙˙
 
.__inference_critic_model_layer_call_fn_3073559V	
0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ź
D__inference_dense_2_layer_call_and_return_conditional_losses_3073605d	
3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙
Ş ")˘&

0˙˙˙˙˙˙˙˙˙2
 
)__inference_dense_2_layer_call_fn_3073614W	
3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙2Ź
D__inference_dense_3_layer_call_and_return_conditional_losses_3073644d3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙2
Ş ")˘&

0˙˙˙˙˙˙˙˙˙
 
)__inference_dense_3_layer_call_fn_3073653W3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙2
Ş "˙˙˙˙˙˙˙˙˙Ľ
%__inference_signature_wrapper_3073574|	
;˘8
˘ 
1Ş.
,
input_1!
input_1˙˙˙˙˙˙˙˙˙"7Ş4
2
output_1&#
output_1˙˙˙˙˙˙˙˙˙