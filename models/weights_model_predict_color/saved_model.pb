ΆΧ
Μ’
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68έ

my_model_save_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	°	*/
shared_name my_model_save_1/dense_6/kernel

2my_model_save_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpmy_model_save_1/dense_6/kernel*
_output_shapes
:	°	*
dtype0

my_model_save_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namemy_model_save_1/dense_6/bias

0my_model_save_1/dense_6/bias/Read/ReadVariableOpReadVariableOpmy_model_save_1/dense_6/bias*
_output_shapes
:*
dtype0

my_model_save_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name my_model_save_1/dense_7/kernel

2my_model_save_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpmy_model_save_1/dense_7/kernel*
_output_shapes

:*
dtype0

my_model_save_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namemy_model_save_1/dense_7/bias

0my_model_save_1/dense_7/bias/Read/ReadVariableOpReadVariableOpmy_model_save_1/dense_7/bias*
_output_shapes
:*
dtype0

NoOpNoOp
μ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§
valueB B
Ο
dense_layers
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures*


0
1
2*
 
0
1
2
3*
 
0
1
2
3*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
¦

kernel
bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
^X
VARIABLE_VALUEmy_model_save_1/dense_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEmy_model_save_1/dense_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEmy_model_save_1/dense_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEmy_model_save_1/dense_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 


0
1
2*
* 
* 
* 
* 
* 
* 
* 

)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
·
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_save_1/dense_6/kernelmy_model_save_1/dense_6/biasmy_model_save_1/dense_7/kernelmy_model_save_1/dense_7/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1377207
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
μ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2my_model_save_1/dense_6/kernel/Read/ReadVariableOp0my_model_save_1/dense_6/bias/Read/ReadVariableOp2my_model_save_1/dense_7/kernel/Read/ReadVariableOp0my_model_save_1/dense_7/bias/Read/ReadVariableOpConst*
Tin

2*
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
 __inference__traced_save_1377293

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_save_1/dense_6/kernelmy_model_save_1/dense_6/biasmy_model_save_1/dense_7/kernelmy_model_save_1/dense_7/bias*
Tin	
2*
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
#__inference__traced_restore_1377315ή
Θ
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377218

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????°  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????°	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε

)__inference_dense_6_layer_call_fn_1377227

inputs
unknown:	°	
	unknown_0:
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1377070o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????°	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????°	
 
_user_specified_nameinputs
ρ
―
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377094
x"
dense_6_1377071:	°	
dense_6_1377073:!
dense_7_1377088:
dense_7_1377090:
identity’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall·
flatten_3/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377057
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_1377071dense_6_1377073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1377070
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_1377088dense_7_1377090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1377087w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
 

υ
D__inference_dense_7_layer_call_and_return_conditional_losses_1377258

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
²
§
"__inference__wrapped_model_1377044
input_1I
6my_model_save_1_dense_6_matmul_readvariableop_resource:	°	E
7my_model_save_1_dense_6_biasadd_readvariableop_resource:H
6my_model_save_1_dense_7_matmul_readvariableop_resource:E
7my_model_save_1_dense_7_biasadd_readvariableop_resource:
identity’.my_model_save_1/dense_6/BiasAdd/ReadVariableOp’-my_model_save_1/dense_6/MatMul/ReadVariableOp’.my_model_save_1/dense_7/BiasAdd/ReadVariableOp’-my_model_save_1/dense_7/MatMul/ReadVariableOpp
my_model_save_1/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????°  
!my_model_save_1/flatten_3/ReshapeReshapeinput_1(my_model_save_1/flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????°	₯
-my_model_save_1/dense_6/MatMul/ReadVariableOpReadVariableOp6my_model_save_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	°	*
dtype0½
my_model_save_1/dense_6/MatMulMatMul*my_model_save_1/flatten_3/Reshape:output:05my_model_save_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????’
.my_model_save_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp7my_model_save_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
my_model_save_1/dense_6/BiasAddBiasAdd(my_model_save_1/dense_6/MatMul:product:06my_model_save_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_model_save_1/dense_6/TanhTanh(my_model_save_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????€
-my_model_save_1/dense_7/MatMul/ReadVariableOpReadVariableOp6my_model_save_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0³
my_model_save_1/dense_7/MatMulMatMul my_model_save_1/dense_6/Tanh:y:05my_model_save_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????’
.my_model_save_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp7my_model_save_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
my_model_save_1/dense_7/BiasAddBiasAdd(my_model_save_1/dense_7/MatMul:product:06my_model_save_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_model_save_1/dense_7/SoftmaxSoftmax(my_model_save_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
IdentityIdentity)my_model_save_1/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp/^my_model_save_1/dense_6/BiasAdd/ReadVariableOp.^my_model_save_1/dense_6/MatMul/ReadVariableOp/^my_model_save_1/dense_7/BiasAdd/ReadVariableOp.^my_model_save_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2`
.my_model_save_1/dense_6/BiasAdd/ReadVariableOp.my_model_save_1/dense_6/BiasAdd/ReadVariableOp2^
-my_model_save_1/dense_6/MatMul/ReadVariableOp-my_model_save_1/dense_6/MatMul/ReadVariableOp2`
.my_model_save_1/dense_7/BiasAdd/ReadVariableOp.my_model_save_1/dense_7/BiasAdd/ReadVariableOp2^
-my_model_save_1/dense_7/MatMul/ReadVariableOp-my_model_save_1/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1


φ
D__inference_dense_6_layer_call_and_return_conditional_losses_1377238

inputs1
matmul_readvariableop_resource:	°	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	°	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????°	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????°	
 
_user_specified_nameinputs
¨
Π
1__inference_my_model_save_1_layer_call_fn_1377172
x
unknown:	°	
	unknown_0:
	unknown_1:
	unknown_2:
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
Β

)__inference_dense_7_layer_call_fn_1377247

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1377087o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs


φ
D__inference_dense_6_layer_call_and_return_conditional_losses_1377070

inputs1
matmul_readvariableop_resource:	°	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	°	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????°	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????°	
 
_user_specified_nameinputs
Θ
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377057

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????°  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????°	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
Λ
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377192
x9
&dense_6_matmul_readvariableop_resource:	°	5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity’dense_6/BiasAdd/ReadVariableOp’dense_6/MatMul/ReadVariableOp’dense_7/BiasAdd/ReadVariableOp’dense_7/MatMul/ReadVariableOp`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????°  l
flatten_3/ReshapeReshapexflatten_3/Const:output:0*
T0*(
_output_shapes
:?????????°	
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	°	*
dtype0
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldense_6/Tanh:y:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Θ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
³
G
+__inference_flatten_3_layer_call_fn_1377212

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377057a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs

΅
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377159
input_1"
dense_6_1377148:	°	
dense_6_1377150:!
dense_7_1377153:
dense_7_1377155:
identity’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall½
flatten_3/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377057
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_1377148dense_6_1377150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1377070
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_1377153dense_7_1377155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1377087w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
 

υ
D__inference_dense_7_layer_call_and_return_conditional_losses_1377087

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ρ
ε
 __inference__traced_save_1377293
file_prefix=
9savev2_my_model_save_1_dense_6_kernel_read_readvariableop;
7savev2_my_model_save_1_dense_6_bias_read_readvariableop=
9savev2_my_model_save_1_dense_7_kernel_read_readvariableop;
7savev2_my_model_save_1_dense_7_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ͺ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Σ
valueΙBΖB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_my_model_save_1_dense_6_kernel_read_readvariableop7savev2_my_model_save_1_dense_6_bias_read_readvariableop9savev2_my_model_save_1_dense_7_kernel_read_readvariableop7savev2_my_model_save_1_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*8
_input_shapes'
%: :	°	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	°	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 

’
#__inference__traced_restore_1377315
file_prefixB
/assignvariableop_my_model_save_1_dense_6_kernel:	°	=
/assignvariableop_1_my_model_save_1_dense_6_bias:C
1assignvariableop_2_my_model_save_1_dense_7_kernel:=
/assignvariableop_3_my_model_save_1_dense_7_bias:

identity_5’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_2’AssignVariableOp_3­
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Σ
valueΙBΖB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ·
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp/assignvariableop_my_model_save_1_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp/assignvariableop_1_my_model_save_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_2AssignVariableOp1assignvariableop_2_my_model_save_1_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp/assignvariableop_3_my_model_save_1_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¬

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ί
Φ
1__inference_my_model_save_1_layer_call_fn_1377105
input_1
unknown:	°	
	unknown_0:
	unknown_1:
	unknown_2:
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1

Κ
%__inference_signature_wrapper_1377207
input_1
unknown:	°	
	unknown_0:
	unknown_1:
	unknown_2:
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1377044o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:=
δ
dense_layers
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures"
_tf_keras_model
5

0
1
2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
1__inference_my_model_save_1_layer_call_fn_1377105
1__inference_my_model_save_1_layer_call_fn_1377172?
₯²‘
FullArgSpec$
args
jself
jx

jdata_aug
varargs
 
varkw
 
defaults’
p

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Π2Ν
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377192
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377159?
₯²‘
FullArgSpec$
args
jself
jx

jdata_aug
varargs
 
varkw
 
defaults’
p

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ΝBΚ
"__inference__wrapped_model_1377044input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
serving_default"
signature_map
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
1:/	°	2my_model_save_1/dense_6/kernel
*:(2my_model_save_1/dense_6/bias
0:.2my_model_save_1/dense_7/kernel
*:(2my_model_save_1/dense_7/bias
 "
trackable_list_wrapper
5

0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΜBΙ
%__inference_signature_wrapper_1377207input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_flatten_3_layer_call_fn_1377212’
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
annotationsͺ *
 
π2ν
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377218’
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
annotationsͺ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_dense_6_layer_call_fn_1377227’
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
annotationsͺ *
 
ξ2λ
D__inference_dense_6_layer_call_and_return_conditional_losses_1377238’
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
annotationsͺ *
 
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
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_dense_7_layer_call_fn_1377247’
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
annotationsͺ *
 
ξ2λ
D__inference_dense_7_layer_call_and_return_conditional_losses_1377258’
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
annotationsͺ *
 
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"__inference__wrapped_model_1377044u8’5
.’+
)&
input_1?????????
ͺ "3ͺ0
.
output_1"
output_1?????????₯
D__inference_dense_6_layer_call_and_return_conditional_losses_1377238]0’-
&’#
!
inputs?????????°	
ͺ "%’"

0?????????
 }
)__inference_dense_6_layer_call_fn_1377227P0’-
&’#
!
inputs?????????°	
ͺ "?????????€
D__inference_dense_7_layer_call_and_return_conditional_losses_1377258\/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 |
)__inference_dense_7_layer_call_fn_1377247O/’,
%’"
 
inputs?????????
ͺ "?????????«
F__inference_flatten_3_layer_call_and_return_conditional_losses_1377218a7’4
-’*
(%
inputs?????????
ͺ "&’#

0?????????°	
 
+__inference_flatten_3_layer_call_fn_1377212T7’4
-’*
(%
inputs?????????
ͺ "?????????°	»
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377159k<’9
2’/
)&
input_1?????????
p
ͺ "%’"

0?????????
 ΅
L__inference_my_model_save_1_layer_call_and_return_conditional_losses_1377192e6’3
,’)
# 
x?????????
p
ͺ "%’"

0?????????
 
1__inference_my_model_save_1_layer_call_fn_1377105^<’9
2’/
)&
input_1?????????
p
ͺ "?????????
1__inference_my_model_save_1_layer_call_fn_1377172X6’3
,’)
# 
x?????????
p
ͺ "?????????ͺ
%__inference_signature_wrapper_1377207C’@
’ 
9ͺ6
4
input_1)&
input_1?????????"3ͺ0
.
output_1"
output_1?????????