ę
Ż£
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
¾
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
 "serve*2.4.0-dev202007112v1.12.1-36285-g51be86b23b8Ł
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
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
Ŗ
'RMSprop/critic_model/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'RMSprop/critic_model/dense_2/kernel/rms
£
;RMSprop/critic_model/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOp'RMSprop/critic_model/dense_2/kernel/rms*
_output_shapes

:2*
dtype0
¢
%RMSprop/critic_model/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%RMSprop/critic_model/dense_2/bias/rms

9RMSprop/critic_model/dense_2/bias/rms/Read/ReadVariableOpReadVariableOp%RMSprop/critic_model/dense_2/bias/rms*
_output_shapes
:2*
dtype0
Ŗ
'RMSprop/critic_model/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'RMSprop/critic_model/dense_3/kernel/rms
£
;RMSprop/critic_model/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOp'RMSprop/critic_model/dense_3/kernel/rms*
_output_shapes

:2*
dtype0
¢
%RMSprop/critic_model/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%RMSprop/critic_model/dense_3/bias/rms

9RMSprop/critic_model/dense_3/bias/rms/Read/ReadVariableOpReadVariableOp%RMSprop/critic_model/dense_3/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
°
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ė
valueįBŽ B×


dense1
	value
opt
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
k
iter
	decay
learning_rate
momentum
rho		rms)	
rms*	rms+	rms,

	0

1
2
3

	0

1
2
3
 
­
trainable_variables
metrics

layers
layer_metrics
	variables
non_trainable_variables
regularization_losses
layer_regularization_losses
 
YW
VARIABLE_VALUEcritic_model/dense_2/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcritic_model/dense_2/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
­
trainable_variables
metrics

 layers
!layer_metrics
	variables
"non_trainable_variables
regularization_losses
#layer_regularization_losses
XV
VARIABLE_VALUEcritic_model/dense_3/kernel'value/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcritic_model/dense_3/bias%value/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
$metrics

%layers
&layer_metrics
	variables
'non_trainable_variables
regularization_losses
(layer_regularization_losses
EC
VARIABLE_VALUERMSprop/iter#opt/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUERMSprop/decay$opt/decay/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUERMSprop/learning_rate,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/momentum'opt/momentum/.ATTRIBUTES/VARIABLE_VALUE
CA
VARIABLE_VALUERMSprop/rho"opt/rho/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUE'RMSprop/critic_model/dense_2/kernel/rms@dense1/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE%RMSprop/critic_model/dense_2/bias/rms>dense1/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'RMSprop/critic_model/dense_3/kernel/rms?value/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE%RMSprop/critic_model/dense_3/bias/rms=value/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
®
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1critic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_922662
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/critic_model/dense_2/kernel/Read/ReadVariableOp-critic_model/dense_2/bias/Read/ReadVariableOp/critic_model/dense_3/kernel/Read/ReadVariableOp-critic_model/dense_3/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp;RMSprop/critic_model/dense_2/kernel/rms/Read/ReadVariableOp9RMSprop/critic_model/dense_2/bias/rms/Read/ReadVariableOp;RMSprop/critic_model/dense_3/kernel/rms/Read/ReadVariableOp9RMSprop/critic_model/dense_3/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_922803

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rho'RMSprop/critic_model/dense_2/kernel/rms%RMSprop/critic_model/dense_2/bias/rms'RMSprop/critic_model/dense_3/kernel/rms%RMSprop/critic_model/dense_3/bias/rms*
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_922852’Ż
¶
ü
H__inference_critic_model_layer_call_and_return_conditional_losses_922633
input_1
dense_2_922581
dense_2_922583
dense_3_922627
dense_3_922629
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallb
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
:’’’’’’’’’2

ExpandDims 
dense_2/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0dense_2_922581dense_2_922583*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_9225702!
dense_2/StatefulPartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_922627dense_3_922629*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_9226162!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
’
®
C__inference_dense_3_layer_call_and_return_conditional_losses_922732

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
:’’’’’’’’’22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ß
®
C__inference_dense_2_layer_call_and_return_conditional_losses_922570

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
:’’’’’’’’’2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’:::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
®
C__inference_dense_3_layer_call_and_return_conditional_losses_922616

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
:’’’’’’’’’22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs


$__inference_signature_wrapper_922662
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_9225332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ź
}
(__inference_dense_2_layer_call_fn_922702

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_9225702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
į:
ź
"__inference__traced_restore_922852
file_prefix0
,assignvariableop_critic_model_dense_2_kernel0
,assignvariableop_1_critic_model_dense_2_bias2
.assignvariableop_2_critic_model_dense_3_kernel0
,assignvariableop_3_critic_model_dense_3_bias#
assignvariableop_4_rmsprop_iter$
 assignvariableop_5_rmsprop_decay,
(assignvariableop_6_rmsprop_learning_rate'
#assignvariableop_7_rmsprop_momentum"
assignvariableop_8_rmsprop_rho>
:assignvariableop_9_rmsprop_critic_model_dense_2_kernel_rms=
9assignvariableop_10_rmsprop_critic_model_dense_2_bias_rms?
;assignvariableop_11_rmsprop_critic_model_dense_3_kernel_rms=
9assignvariableop_12_rmsprop_critic_model_dense_3_bias_rms
identity_14¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB'opt/momentum/.ATTRIBUTES/VARIABLE_VALUEB"opt/rho/.ATTRIBUTES/VARIABLE_VALUEB@dense1/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB>dense1/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB?value/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB=value/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesń
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_critic_model_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_critic_model_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_critic_model_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_critic_model_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5„
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ø
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9æ
AssignVariableOp_9AssignVariableOp:assignvariableop_9_rmsprop_critic_model_dense_2_kernel_rmsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Į
AssignVariableOp_10AssignVariableOp9assignvariableop_10_rmsprop_critic_model_dense_2_bias_rmsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ć
AssignVariableOp_11AssignVariableOp;assignvariableop_11_rmsprop_critic_model_dense_3_kernel_rmsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Į
AssignVariableOp_12AssignVariableOp9assignvariableop_12_rmsprop_critic_model_dense_3_bias_rmsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpü
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13ļ
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
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
ź
}
(__inference_dense_3_layer_call_fn_922741

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_9226162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ß
®
C__inference_dense_2_layer_call_and_return_conditional_losses_922693

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
:’’’’’’’’’2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’:::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ĄW
­
!__inference__wrapped_model_922533
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
critic_model/ExpandDims/dim„
critic_model/ExpandDims
ExpandDimsinput_1$critic_model/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
critic_model/ExpandDimsÕ
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
,critic_model/dense_2/Tensordot/GatherV2/axisŗ
'critic_model/dense_2/Tensordot/GatherV2GatherV2-critic_model/dense_2/Tensordot/Shape:output:0,critic_model/dense_2/Tensordot/free:output:05critic_model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/GatherV2¢
.critic_model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_2/Tensordot/GatherV2_1/axisĄ
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
$critic_model/dense_2/Tensordot/ConstŌ
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
%critic_model/dense_2/Tensordot/concatą
$critic_model/dense_2/Tensordot/stackPack,critic_model/dense_2/Tensordot/Prod:output:0.critic_model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_2/Tensordot/stacké
(critic_model/dense_2/Tensordot/transpose	Transpose critic_model/ExpandDims:output:0.critic_model/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’2*
(critic_model/dense_2/Tensordot/transposeó
&critic_model/dense_2/Tensordot/ReshapeReshape,critic_model/dense_2/Tensordot/transpose:y:0-critic_model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2(
&critic_model/dense_2/Tensordot/Reshapeņ
%critic_model/dense_2/Tensordot/MatMulMatMul/critic_model/dense_2/Tensordot/Reshape:output:05critic_model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22'
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
,critic_model/dense_2/Tensordot/concat_1/axis¦
'critic_model/dense_2/Tensordot/concat_1ConcatV20critic_model/dense_2/Tensordot/GatherV2:output:0/critic_model/dense_2/Tensordot/Const_2:output:05critic_model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/concat_1ä
critic_model/dense_2/TensordotReshape/critic_model/dense_2/Tensordot/MatMul:product:00critic_model/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22 
critic_model/dense_2/TensordotĖ
+critic_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+critic_model/dense_2/BiasAdd/ReadVariableOpŪ
critic_model/dense_2/BiasAddBiasAdd'critic_model/dense_2/Tensordot:output:03critic_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22
critic_model/dense_2/BiasAdd
critic_model/dense_2/ReluRelu%critic_model/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
critic_model/dense_2/ReluÕ
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
#critic_model/dense_3/Tensordot/free£
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
,critic_model/dense_3/Tensordot/GatherV2/axisŗ
'critic_model/dense_3/Tensordot/GatherV2GatherV2-critic_model/dense_3/Tensordot/Shape:output:0,critic_model/dense_3/Tensordot/free:output:05critic_model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/GatherV2¢
.critic_model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_3/Tensordot/GatherV2_1/axisĄ
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
$critic_model/dense_3/Tensordot/ConstŌ
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
%critic_model/dense_3/Tensordot/concatą
$critic_model/dense_3/Tensordot/stackPack,critic_model/dense_3/Tensordot/Prod:output:0.critic_model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_3/Tensordot/stackš
(critic_model/dense_3/Tensordot/transpose	Transpose'critic_model/dense_2/Relu:activations:0.critic_model/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’22*
(critic_model/dense_3/Tensordot/transposeó
&critic_model/dense_3/Tensordot/ReshapeReshape,critic_model/dense_3/Tensordot/transpose:y:0-critic_model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2(
&critic_model/dense_3/Tensordot/Reshapeņ
%critic_model/dense_3/Tensordot/MatMulMatMul/critic_model/dense_3/Tensordot/Reshape:output:05critic_model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2'
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
,critic_model/dense_3/Tensordot/concat_1/axis¦
'critic_model/dense_3/Tensordot/concat_1ConcatV20critic_model/dense_3/Tensordot/GatherV2:output:0/critic_model/dense_3/Tensordot/Const_2:output:05critic_model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/concat_1ä
critic_model/dense_3/TensordotReshape/critic_model/dense_3/Tensordot/MatMul:product:00critic_model/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2 
critic_model/dense_3/TensordotĖ
+critic_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+critic_model/dense_3/BiasAdd/ReadVariableOpŪ
critic_model/dense_3/BiasAddBiasAdd'critic_model/dense_3/Tensordot:output:03critic_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2
critic_model/dense_3/BiasAdd}
IdentityIdentity%critic_model/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ä'
ā
__inference__traced_save_922803
file_prefix:
6savev2_critic_model_dense_2_kernel_read_readvariableop8
4savev2_critic_model_dense_2_bias_read_readvariableop:
6savev2_critic_model_dense_3_kernel_read_readvariableop8
4savev2_critic_model_dense_3_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopF
Bsavev2_rmsprop_critic_model_dense_2_kernel_rms_read_readvariableopD
@savev2_rmsprop_critic_model_dense_2_bias_rms_read_readvariableopF
Bsavev2_rmsprop_critic_model_dense_3_kernel_rms_read_readvariableopD
@savev2_rmsprop_critic_model_dense_3_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
value3B1 B+_temp_44af2f634f624fc4bc5378bc573888fc/part2	
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB'opt/momentum/.ATTRIBUTES/VARIABLE_VALUEB"opt/rho/.ATTRIBUTES/VARIABLE_VALUEB@dense1/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB>dense1/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB?value/kernel/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB=value/bias/.OPTIMIZER_SLOT/opt/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_critic_model_dense_2_kernel_read_readvariableop4savev2_critic_model_dense_2_bias_read_readvariableop6savev2_critic_model_dense_3_kernel_read_readvariableop4savev2_critic_model_dense_3_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableopBsavev2_rmsprop_critic_model_dense_2_kernel_rms_read_readvariableop@savev2_rmsprop_critic_model_dense_2_bias_rms_read_readvariableopBsavev2_rmsprop_critic_model_dense_3_kernel_rms_read_readvariableop@savev2_rmsprop_critic_model_dense_3_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
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

identity_1Identity_1:output:0*a
_input_shapesP
N: :2:2:2:: : : : : :2:2:2:: 2(
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
: :$
 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
°
”
-__inference_critic_model_layer_call_fn_922647
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_critic_model_layer_call_and_return_conditional_losses_9226332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1"ŹL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Æ
serving_default
;
input_10
serving_default_input_1:0’’’’’’’’’@
output_14
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ū?
ą

dense1
	value
opt
trainable_variables
	variables
regularization_losses
	keras_api

signatures
-_default_save_signature
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_modelź{"class_name": "CriticModel", "name": "critic_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CriticModel"}}
ī

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
0__call__
*1&call_and_return_all_conditional_losses"É
_tf_keras_layerÆ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [45, 1, 1]}}
ņ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [45, 1, 50]}}
~
iter
	decay
learning_rate
momentum
rho		rms)	
rms*	rms+	rms,"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
trainable_variables
metrics

layers
layer_metrics
	variables
non_trainable_variables
regularization_losses
layer_regularization_losses
.__call__
-_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
4serving_default"
signature_map
-:+22critic_model/dense_2/kernel
':%22critic_model/dense_2/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
metrics

 layers
!layer_metrics
	variables
"non_trainable_variables
regularization_losses
#layer_regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
-:+22critic_model/dense_3/kernel
':%2critic_model/dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
$metrics

%layers
&layer_metrics
	variables
'non_trainable_variables
regularization_losses
(layer_regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
.
0
1"
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
 "
trackable_list_wrapper
7:522'RMSprop/critic_model/dense_2/kernel/rms
1:/22%RMSprop/critic_model/dense_2/bias/rms
7:522'RMSprop/critic_model/dense_3/kernel/rms
1:/2%RMSprop/critic_model/dense_3/bias/rms
ß2Ü
!__inference__wrapped_model_922533¶
²
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
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ū2ų
-__inference_critic_model_layer_call_fn_922647Ę
²
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
annotationsŖ *&¢#
!
input_1’’’’’’’’’
2
H__inference_critic_model_layer_call_and_return_conditional_losses_922633Ę
²
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
annotationsŖ *&¢#
!
input_1’’’’’’’’’
Ņ2Ļ
(__inference_dense_2_layer_call_fn_922702¢
²
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
annotationsŖ *
 
ķ2ź
C__inference_dense_2_layer_call_and_return_conditional_losses_922693¢
²
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
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_3_layer_call_fn_922741¢
²
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
annotationsŖ *
 
ķ2ź
C__inference_dense_3_layer_call_and_return_conditional_losses_922732¢
²
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
annotationsŖ *
 
3B1
$__inference_signature_wrapper_922662input_1
!__inference__wrapped_model_922533q	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "7Ŗ4
2
output_1&#
output_1’’’’’’’’’Æ
H__inference_critic_model_layer_call_and_return_conditional_losses_922633c	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’
 
-__inference_critic_model_layer_call_fn_922647V	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "’’’’’’’’’«
C__inference_dense_2_layer_call_and_return_conditional_losses_922693d	
3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’2
 
(__inference_dense_2_layer_call_fn_922702W	
3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’2«
C__inference_dense_3_layer_call_and_return_conditional_losses_922732d3¢0
)¢&
$!
inputs’’’’’’’’’2
Ŗ ")¢&

0’’’’’’’’’
 
(__inference_dense_3_layer_call_fn_922741W3¢0
)¢&
$!
inputs’’’’’’’’’2
Ŗ "’’’’’’’’’¤
$__inference_signature_wrapper_922662|	
;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"7Ŗ4
2
output_1&#
output_1’’’’’’’’’