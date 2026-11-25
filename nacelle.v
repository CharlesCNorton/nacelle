(* Nacelle's proofs, in Rocq.  Signature (-,+,+,+), G = c = 1.
   1. Energy conditions of a Hawking-Ellis type I tensor, and the multiplier test for the null condition.
   2. The Maxwell stress tensor.
   3-4. The flat-slice theorem, pointwise and integrated.
   5. Energy conditions from multiplier certificates.
   6. The face theorem and the certificates of the closed-form drag and rocket shells, which nacelle shells
      certify writes in place of everything from its heading to the end of the file. *)

From Stdlib Require Import Reals Lra Psatz Ring.
From Coquelicot Require Import Coquelicot.
From Interval Require Import Tactic.

Local Open Scope R_scope.

(* 1. Hawking-Ellis type I: energy conditions as inequalities on (rho, p1, p2, p3). *)

Section TypeI.
Variables rho p1 p2 p3 : R.

(* T(k,k) in the orthonormal frame that diagonalises T. *)
Definition Tdiag (k0 k1 k2 k3 : R) : R :=
  rho*(k0*k0) + p1*(k1*k1) + p2*(k2*k2) + p3*(k3*k3).

Lemma Tdiag_split k0 k1 k2 k3 :
  Tdiag k0 k1 k2 k3 =
    rho*(k0*k0 - (k1*k1 + k2*k2 + k3*k3))
    + (rho+p1)*(k1*k1) + (rho+p2)*(k2*k2) + (rho+p3)*(k3*k3).
Proof. unfold Tdiag; ring. Qed.

Theorem typeI_NEC :
  (forall k0 k1 k2 k3, k0*k0 = k1*k1 + k2*k2 + k3*k3 -> 0 <= Tdiag k0 k1 k2 k3)
  <-> 0 <= rho+p1 /\ 0 <= rho+p2 /\ 0 <= rho+p3.
Proof.
  split.
  - intros H.
    pose proof (H 1 1 0 0 ltac:(ring)) as H1.
    pose proof (H 1 0 1 0 ltac:(ring)) as H2.
    pose proof (H 1 0 0 1 ltac:(ring)) as H3.
    unfold Tdiag in H1, H2, H3. lra.
  - intros (H1 & H2 & H3) k0 k1 k2 k3 Hk.
    rewrite Tdiag_split, Hk.
    pose proof (Rmult_le_pos _ _ H1 (Rle_0_sqr k1)).
    pose proof (Rmult_le_pos _ _ H2 (Rle_0_sqr k2)).
    pose proof (Rmult_le_pos _ _ H3 (Rle_0_sqr k3)).
    unfold Rsqr in *. nra.
Qed.

(* Causal vectors; the timelike statement follows by continuity. *)
Theorem typeI_WEC :
  (forall k0 k1 k2 k3, k1*k1 + k2*k2 + k3*k3 <= k0*k0 -> 0 <= Tdiag k0 k1 k2 k3)
  <-> 0 <= rho /\ 0 <= rho+p1 /\ 0 <= rho+p2 /\ 0 <= rho+p3.
Proof.
  split.
  - intros H.
    pose proof (H 1 0 0 0 ltac:(lra)) as H0.
    pose proof (H 1 1 0 0 ltac:(lra)) as H1.
    pose proof (H 1 0 1 0 ltac:(lra)) as H2.
    pose proof (H 1 0 0 1 ltac:(lra)) as H3.
    unfold Tdiag in H0, H1, H2, H3. lra.
  - intros (H0 & H1 & H2 & H3) k0 k1 k2 k3 Hk.
    rewrite Tdiag_split.
    assert (0 <= rho*(k0*k0 - (k1*k1 + k2*k2 + k3*k3)))
      by (apply Rmult_le_pos; lra).
    pose proof (Rmult_le_pos _ _ H1 (Rle_0_sqr k1)).
    pose proof (Rmult_le_pos _ _ H2 (Rle_0_sqr k2)).
    pose proof (Rmult_le_pos _ _ H3 (Rle_0_sqr k3)).
    unfold Rsqr in *. lra.
Qed.

(* Flux -T^mu_nu k^nu = (rho k0, -p1 k1, -p2 k2, -p3 k3) is causal for causal k. *)
Theorem typeI_DEC_flux :
  (forall k0 k1 k2 k3, k1*k1 + k2*k2 + k3*k3 <= k0*k0 ->
     (p1*k1)*(p1*k1) + (p2*k2)*(p2*k2) + (p3*k3)*(p3*k3) <= (rho*k0)*(rho*k0))
  <-> p1*p1 <= rho*rho /\ p2*p2 <= rho*rho /\ p3*p3 <= rho*rho.
Proof.
  split.
  - intros H.
    pose proof (H 1 1 0 0 ltac:(lra)) as H1.
    pose proof (H 1 0 1 0 ltac:(lra)) as H2.
    pose proof (H 1 0 0 1 ltac:(lra)) as H3.
    nra.
  - intros (H1 & H2 & H3) k0 k1 k2 k3 Hk.
    pose proof (Rle_0_sqr k1) as S1.
    pose proof (Rle_0_sqr k2) as S2.
    pose proof (Rle_0_sqr k3) as S3.
    pose proof (Rle_0_sqr rho) as S0.
    unfold Rsqr in *.
    pose proof (Rmult_le_compat_r _ _ _ S1 H1).
    pose proof (Rmult_le_compat_r _ _ _ S2 H2).
    pose proof (Rmult_le_compat_r _ _ _ S3 H3).
    pose proof (Rmult_le_compat_l _ _ _ S0 Hk).
    nra.
Qed.
End TypeI.

(* 1b. A multiplier lam with T + lam * eta positive semidefinite certifies the null
       energy condition at a point, for an arbitrary symmetric T in an orthonormal frame. *)

Section Multiplier.
Variables t00 t01 t02 t03 t11 t12 t13 t22 t23 t33 lam : R.

Definition Tquad (k0 k1 k2 k3 : R) : R :=
    t00*(k0*k0) + t11*(k1*k1) + t22*(k2*k2) + t33*(k3*k3)
  + 2*(t01*(k0*k1) + t02*(k0*k2) + t03*(k0*k3)
       + t12*(k1*k2) + t13*(k1*k3) + t23*(k2*k3)).

Definition eta (k0 k1 k2 k3 : R) : R := - (k0*k0) + k1*k1 + k2*k2 + k3*k3.

Theorem multiplier_certifies_NEC :
  (forall k0 k1 k2 k3, 0 <= Tquad k0 k1 k2 k3 + lam * eta k0 k1 k2 k3) ->
  forall k0 k1 k2 k3, k0*k0 = k1*k1 + k2*k2 + k3*k3 -> 0 <= Tquad k0 k1 k2 k3.
Proof.
  intros H k0 k1 k2 k3 Hk.
  pose proof (H k0 k1 k2 k3) as Hq.
  assert (eta k0 k1 k2 k3 = 0) as E by (unfold eta; lra).
  rewrite E in Hq. lra.
Qed.
End Multiplier.

(* 2. The Maxwell stress tensor obeys the null and dominant energy conditions identically. *)

Section Maxwell.
Variables e1 e2 e3 b1 b2 b3 : R.

Definition E2 := e1*e1 + e2*e2 + e3*e3.
Definition B2 := b1*b1 + b2*b2 + b3*b3.
Definition EB := e1*b1 + e2*b2 + e3*b3.
Definition u := (E2 + B2) / 2.          (* T^00 *)
Definition P1 := e2*b3 - e3*b2.         (* T^0i = (E x B)_i *)
Definition P2 := e3*b1 - e1*b3.
Definition P3 := e1*b2 - e2*b1.

Lemma maxwell_energy_nonneg : 0 <= u.
Proof. unfold u, E2, B2. nra. Qed.

Lemma maxwell_rainich :
  u*u - (P1*P1 + P2*P2 + P3*P3) = ((E2 - B2)/2)*((E2 - B2)/2) + EB*EB.
Proof. unfold u, P1, P2, P3, E2, B2, EB. field. Qed.

Theorem maxwell_DEC_flux : P1*P1 + P2*P2 + P3*P3 <= u*u.
Proof.
  pose proof maxwell_rainich as H.
  pose proof (Rle_0_sqr ((E2 - B2)/2)) as Ha.
  pose proof (Rle_0_sqr EB) as Hb.
  unfold Rsqr in *. lra.
Qed.

Section Null.
Variables n1 n2 n3 : R.

Definition En := e1*n1 + e2*n2 + e3*n3.
Definition Bn := b1*n1 + b2*n2 + b3*n3.

(* T(k,k) for k = (1, n), with T_ij = u delta_ij - E_i E_j - B_i B_j. *)
Definition Tkk :=
  u - 2*(P1*n1 + P2*n2 + P3*n3)
  + (u*(n1*n1 + n2*n2 + n3*n3) - En*En - Bn*Bn).

(* V = E - (E.n) n + n x B *)
Definition V1 := e1 - En*n1 + (n2*b3 - n3*b2).
Definition V2 := e2 - En*n2 + (n3*b1 - n1*b3).
Definition V3 := e3 - En*n3 + (n1*b2 - n2*b1).

Lemma maxwell_Tkk_square :
  Tkk = V1*V1 + V2*V2 + V3*V3
        + (1 - (n1*n1 + n2*n2 + n3*n3)) * ((B2 - E2)/2 + En*En).
Proof. unfold Tkk, V1, V2, V3, u, P1, P2, P3, En, Bn, E2, B2. field. Qed.

Theorem maxwell_NEC : n1*n1 + n2*n2 + n3*n3 = 1 -> 0 <= Tkk.
Proof.
  intros Hn. rewrite maxwell_Tkk_square, Hn.
  pose proof (Rle_0_sqr V1) as H1.
  pose proof (Rle_0_sqr V2) as H2.
  pose proof (Rle_0_sqr V3) as H3.
  unfold Rsqr in *. lra.
Qed.
End Null.
End Maxwell.

(* 3. Flat spatial slices, pointwise. aij stands for d_i beta_j at one point,
      K_ij = (d_i beta_j + d_j beta_i) / (2 alpha), and ham = K^2 - K_ij K^ij,
      which the Hamiltonian constraint equates to 16 pi rho when R3 = 0. *)

Section FlatSlicesPointwise.
Variables a11 a12 a13 a21 a22 a23 a31 a32 a33 alpha : R.
Hypothesis alpha_pos : 0 < alpha.

Definition K (x y : R) := (x + y) / (2*alpha).
Definition trK := K a11 a11 + K a22 a22 + K a33 a33.
Definition KK :=
    K a11 a11 * K a11 a11 + K a12 a21 * K a12 a21 + K a13 a31 * K a13 a31
  + K a21 a12 * K a21 a12 + K a22 a22 * K a22 a22 + K a23 a32 * K a23 a32
  + K a31 a13 * K a31 a13 + K a32 a23 * K a32 a23 + K a33 a33 * K a33 a33.
Definition ham := trK*trK - KK.

Definition minors :=
  2*(a11*a22 - a12*a21 + a11*a33 - a13*a31 + a22*a33 - a23*a32).
Definition curlsq :=
  (a23 - a32)*(a23 - a32) + (a31 - a13)*(a31 - a13) + (a12 - a21)*(a12 - a21).

Lemma ham_flat : alpha*alpha*ham = minors - curlsq/2.
Proof. unfold ham, trK, KK, K, minors, curlsq. field. lra. Qed.

(* Alcubierre: beta = (beta_x, 0, 0), any lapse. *)
Lemma alcubierre_density :
  a12 = 0 -> a13 = 0 -> a22 = 0 -> a23 = 0 -> a32 = 0 -> a33 = 0 ->
  alpha*alpha*ham = - (a21*a21 + a31*a31) / 2.
Proof.
  intros H12 H13 H22 H23 H32 H33.
  rewrite ham_flat. unfold minors, curlsq.
  rewrite H12, H13, H22, H23, H32, H33. field.
Qed.

Corollary alcubierre_nonpositive :
  a12 = 0 -> a13 = 0 -> a22 = 0 -> a23 = 0 -> a32 = 0 -> a33 = 0 -> ham <= 0.
Proof.
  intros H12 H13 H22 H23 H32 H33.
  pose proof (alcubierre_density H12 H13 H22 H23 H32 H33) as H.
  assert (0 < alpha*alpha) by nra.
  nra.
Qed.
End FlatSlicesPointwise.

(* 4. Flat spatial slices, integrated. F is a commutative ring of fields with three
      commuting derivations; Cs marks the fields whose total derivatives integrate to zero. *)

Inductive axis := X | Y | Z.

Declare Scope F_scope.

Section FlatSlicesIntegrated.
Variable F : Type.
Variables (zero one : F) (add mul sub : F -> F -> F) (opp : F -> F).
Hypothesis Fth : ring_theory zero one add mul sub opp (@eq F).
Add Ring Fring : Fth.

Notation "x + y" := (add x y) : F_scope.
Notation "x - y" := (sub x y) : F_scope.
Notation "x * y" := (mul x y) : F_scope.
Local Open Scope F_scope.

Variable D : axis -> F -> F.
Hypothesis D_add : forall i f g, D i (f + g) = D i f + D i g.
Hypothesis D_sub : forall i f g, D i (f - g) = D i f - D i g.
Hypothesis D_mul : forall i f g, D i (f * g) = D i f * g + f * D i g.
Hypothesis D_comm : forall i j f, D i (D j f) = D j (D i f).

Variables b1 b2 b3 : F.      (* shift *)

Definition dv := D X b1 + D Y b2 + D Z b3.
Definition W1 := b1 * dv - (b1 * D X b1 + b2 * D Y b1 + b3 * D Z b1).
Definition W2 := b2 * dv - (b1 * D X b2 + b2 * D Y b2 + b3 * D Z b2).
Definition W3 := b3 * dv - (b1 * D X b3 + b2 * D Y b3 + b3 * D Z b3).
Definition divW := D X W1 + D Y W2 + D Z W3.

Definition tra2 :=
    D X b1 * D X b1 + D Y b2 * D Y b2 + D Z b3 * D Z b3
  + (D X b2 * D Y b1 + D Y b1 * D X b2)
  + (D X b3 * D Z b1 + D Z b1 * D X b3)
  + (D Y b3 * D Z b2 + D Z b2 * D Y b3).

Lemma divergence_identity : divW = dv * dv - tra2.
Proof.
  unfold divW, W1, W2, W3, dv, tra2.
  repeat (rewrite D_sub || rewrite D_add || rewrite D_mul).
  rewrite ?(D_comm Y X), ?(D_comm Z X), ?(D_comm Z Y).
  ring.
Qed.

Definition symsq :=
    (D X b1 + D X b1) * (D X b1 + D X b1)
  + (D X b2 + D Y b1) * (D X b2 + D Y b1)
  + (D X b3 + D Z b1) * (D X b3 + D Z b1)
  + (D Y b1 + D X b2) * (D Y b1 + D X b2)
  + (D Y b2 + D Y b2) * (D Y b2 + D Y b2)
  + (D Y b3 + D Z b2) * (D Y b3 + D Z b2)
  + (D Z b1 + D X b3) * (D Z b1 + D X b3)
  + (D Z b2 + D Y b3) * (D Z b2 + D Y b3)
  + (D Z b3 + D Z b3) * (D Z b3 + D Z b3).

Definition c1 := D Y b3 - D Z b2.
Definition c2 := D Z b1 - D X b3.
Definition c3 := D X b2 - D Y b1.
Definition curl2 := c1 * c1 + c2 * c2 + c3 * c3.
Definition two := one + one.
Definition four := two + two.

Lemma hamiltonian_split :
  four * (dv * dv) - symsq = (divW + divW + divW + divW) - (curl2 + curl2).
Proof.
  rewrite divergence_identity.
  unfold symsq, curl2, c1, c2, c3, tra2, dv, four, two. ring.
Qed.

Variable Int : F -> R.
Variable Cs : F -> Prop.
Hypothesis Cs_add : forall f g, Cs f -> Cs g -> Cs (f + g).
Hypothesis Cs_sub : forall f g, Cs f -> Cs g -> Cs (f - g).
Hypothesis Cs_mul : forall f g, Cs f -> Cs (f * g).
Hypothesis Cs_D : forall i f, Cs f -> Cs (D i f).
Hypothesis Int_add : forall f g, Cs f -> Cs g -> Int (f + g) = Rplus (Int f) (Int g).
Hypothesis Int_sub : forall f g, Cs f -> Cs g -> Int (f - g) = Rminus (Int f) (Int g).
Hypothesis Int_D : forall i f, Cs f -> Int (D i f) = R0.
Hypothesis Int_sq : forall f, Rle R0 (Int (f * f)).
Hypotheses (Cs_b1 : Cs b1) (Cs_b2 : Cs b2) (Cs_b3 : Cs b3).

Variables alpha rho16 : F.   (* lapse, and 16 pi rho *)

(* Hamiltonian constraint with R3 = 0, cleared of denominators. *)
Hypothesis hamiltonian :
  four * (alpha * alpha) * rho16 = four * (dv * dv) - symsq.

Lemma Cs_W : Cs W1 /\ Cs W2 /\ Cs W3.
Proof.
  unfold W1, W2, W3. repeat split; apply Cs_sub; repeat apply Cs_add; apply Cs_mul; assumption.
Qed.

Lemma Cs_divW : Cs divW.
Proof.
  destruct Cs_W as (H1 & H2 & H3).
  unfold divW. repeat apply Cs_add; apply Cs_D; assumption.
Qed.

Lemma Int_divW : Int divW = R0.
Proof.
  destruct Cs_W as (H1 & H2 & H3).
  unfold divW.
  rewrite Int_add by auto using Cs_add, Cs_D.
  rewrite Int_add by auto using Cs_D.
  rewrite !Int_D by assumption.
  lra.
Qed.

Lemma Cs_c : Cs c1 /\ Cs c2 /\ Cs c3.
Proof. unfold c1, c2, c3. repeat split; apply Cs_sub; apply Cs_D; assumption. Qed.

Lemma Cs_curl2 : Cs curl2.
Proof.
  destruct Cs_c as (H1 & H2 & H3).
  unfold curl2. repeat apply Cs_add; apply Cs_mul; assumption.
Qed.

Lemma Int_curl2_nonneg : Rle R0 (Int curl2).
Proof.
  destruct Cs_c as (H1 & H2 & H3).
  unfold curl2.
  rewrite Int_add by auto using Cs_add, Cs_mul.
  rewrite Int_add by auto using Cs_mul.
  pose proof (Int_sq c1). pose proof (Int_sq c2). pose proof (Int_sq c3).
  lra.
Qed.

(* Integral of 4 alpha^2 (16 pi rho) = -2 Integral of |curl beta|^2. *)
Theorem flat_slices_integrated_density :
  Int (four * (alpha * alpha) * rho16) = Ropp (Rplus (Int curl2) (Int curl2)).
Proof.
  pose proof Cs_divW as HW. pose proof Cs_curl2 as HC.
  rewrite hamiltonian, hamiltonian_split.
  rewrite Int_sub by auto using Cs_add.
  rewrite !Int_add by auto using Cs_add.
  rewrite Int_divW. lra.
Qed.

Corollary flat_slices_nonpositive :
  Rle (Int (four * (alpha * alpha) * rho16)) R0.
Proof.
  rewrite flat_slices_integrated_density.
  pose proof Int_curl2_nonneg. lra.
Qed.
End FlatSlicesIntegrated.

(* 5. Energy conditions from multiplier certificates.

   T is a symmetric tensor on an orthonormal frame (index 0 timelike) whose components
   T_03, T_13, T_23 vanish, as for an axisymmetric non-rotating configuration:

        E  a  d  0
        a  Sr x  0
        d  x  Th 0
        0  0  0  Ph

   A multiplier mu with T + mu eta positive semidefinite gives the null, weak and strong
   conditions, and with a multiplier lam for which lam eta - T eta T is positive semidefinite
   the dominant condition.  Diagonal dominance or Sylvester's criterion shows the two pencils
   semidefinite. *)

Definition Qf (m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3 : R) : R :=
  m00*(k0*k0) + m11*(k1*k1) + m22*(k2*k2) + m33*(k3*k3)
  + 2*(m01*(k0*k1) + m02*(k0*k2) + m12*(k1*k2)).

Lemma cross_bound n p q : - Rabs n * (p*p + q*q) <= 2*n*(p*q).
Proof.
  pose proof (Rle_0_sqr (p - q)) as H1.
  pose proof (Rle_0_sqr (p + q)) as H2.
  unfold Rsqr in *. unfold Rabs. destruct (Rcase_abs n); nra.
Qed.

Lemma dominant_nonneg m00 m01 m02 m11 m12 m22 m33 :
  Rabs m01 + Rabs m02 <= m00 -> Rabs m01 + Rabs m12 <= m11 ->
  Rabs m02 + Rabs m12 <= m22 -> 0 <= m33 ->
  forall k0 k1 k2 k3, 0 <= Qf m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3.
Proof.
  intros H0 H1 H2 H3 k0 k1 k2 k3. unfold Qf.
  pose proof (cross_bound m01 k0 k1) as C01.
  pose proof (cross_bound m02 k0 k2) as C02.
  pose proof (cross_bound m12 k1 k2) as C12.
  pose proof (Rle_0_sqr k0) as S0. pose proof (Rle_0_sqr k1) as S1.
  pose proof (Rle_0_sqr k2) as S2. pose proof (Rle_0_sqr k3) as S3.
  unfold Rsqr in *.
  assert (0 <= (m00 - Rabs m01 - Rabs m02) * (k0*k0)) by (apply Rmult_le_pos; lra).
  assert (0 <= (m11 - Rabs m01 - Rabs m12) * (k1*k1)) by (apply Rmult_le_pos; lra).
  assert (0 <= (m22 - Rabs m02 - Rabs m12) * (k2*k2)) by (apply Rmult_le_pos; lra).
  assert (0 <= m33 * (k3*k3)) by (apply Rmult_le_pos; lra).
  nra.
Qed.

(* Sylvester's criterion for the same block form: positive leading minors of the 3x3 block and
   a non-negative 3x3 determinant make the form non-negative. *)
Definition minor2 (m00 m01 m11 : R) : R := m00*m11 - m01*m01.
Definition det3 (m00 m01 m02 m11 m12 m22 : R) : R :=
  m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02).

Lemma sylvester_nonneg m00 m01 m02 m11 m12 m22 m33 :
  0 < m00 -> 0 < minor2 m00 m01 m11 -> 0 <= det3 m00 m01 m02 m11 m12 m22 -> 0 <= m33 ->
  forall k0 k1 k2 k3, 0 <= Qf m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3.
Proof.
  intros H0 HA Hd H3 k0 k1 k2 k3.
  unfold minor2, det3 in *.
  assert (Id : m00 * (m00*m11 - m01*m01) * Qf m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3 =
     (m00*m11 - m01*m01) * ((m00*k0 + m01*k1 + m02*k2) * (m00*k0 + m01*k1 + m02*k2))
     + ((m00*m11 - m01*m01)*k1 + (m00*m12 - m01*m02)*k2) * ((m00*m11 - m01*m01)*k1 + (m00*m12 - m01*m02)*k2)
     + (m00 * (m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02))) * (k2*k2)
     + (m00 * (m00*m11 - m01*m01) * m33) * (k3*k3)).
  { unfold Qf. ring. }
  pose proof (Rle_0_sqr (m00*k0 + m01*k1 + m02*k2)) as Sy.
  pose proof (Rle_0_sqr ((m00*m11 - m01*m01)*k1 + (m00*m12 - m01*m02)*k2)) as Sz.
  pose proof (Rle_0_sqr k2) as S2. pose proof (Rle_0_sqr k3) as S3. unfold Rsqr in *.
  assert (0 <= (m00*m11 - m01*m01) * ((m00*k0 + m01*k1 + m02*k2) * (m00*k0 + m01*k1 + m02*k2)))
    by (apply Rmult_le_pos; lra).
  assert (0 <= m00 * (m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02)))
    by (apply Rmult_le_pos; lra).
  assert (0 <= (m00 * (m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02))) * (k2*k2))
    by (apply Rmult_le_pos; lra).
  assert (0 < m00 * (m00*m11 - m01*m01)) as P by (apply Rmult_lt_0_compat; lra).
  assert (0 <= m00 * (m00*m11 - m01*m01) * m33) by (apply Rmult_le_pos; lra).
  assert (0 <= (m00 * (m00*m11 - m01*m01) * m33) * (k3*k3)) by (apply Rmult_le_pos; lra).
  assert (0 <= m00 * (m00*m11 - m01*m01) * Qf m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3) as Q
    by (rewrite Id; lra).
  destruct (Rle_or_lt 0 (Qf m00 m01 m02 m11 m12 m22 m33 k0 k1 k2 k3)) as [Hq | Hq]; [exact Hq |].
  exfalso. pose proof (Rmult_lt_compat_l _ _ _ P Hq) as Hn. rewrite Rmult_0_r in Hn. lra.
Qed.

Section Certificates.
Variables E a d Sr x Th Ph : R.

Definition Tq (k0 k1 k2 k3 : R) : R := Qf E a d Sr x Th Ph k0 k1 k2 k3.
Definition trace : R := - E + Sr + Th + Ph.

(* w_a = T_ab k^b; the energy flux seen by k is (w0, -w1, -w2, -w3) *)
Definition w0 (k0 k1 k2 k3 : R) : R := E*k0 + a*k1 + d*k2.
Definition w1 (k0 k1 k2 k3 : R) : R := a*k0 + Sr*k1 + x*k2.
Definition w2 (k0 k1 k2 k3 : R) : R := d*k0 + x*k1 + Th*k2.
Definition w3 (k0 k1 k2 k3 : R) : R := Ph*k3.

Definition psd_mu (mu : R) : Prop :=
  forall k0 k1 k2 k3, 0 <= Qf (E - mu) a d (Sr + mu) x (Th + mu) (Ph + mu) k0 k1 k2 k3.

Lemma pencil mu k0 k1 k2 k3 :
  Qf (E - mu) a d (Sr + mu) x (Th + mu) (Ph + mu) k0 k1 k2 k3 = Tq k0 k1 k2 k3 + mu * eta k0 k1 k2 k3.
Proof. unfold Tq, Qf, eta. ring. Qed.

Theorem null_condition mu : psd_mu mu ->
  forall k0 k1 k2 k3, eta k0 k1 k2 k3 = 0 -> 0 <= Tq k0 k1 k2 k3.
Proof.
  intros H k0 k1 k2 k3 Hk. specialize (H k0 k1 k2 k3).
  rewrite pencil, Hk in H. lra.
Qed.

Theorem weak_condition mu : psd_mu mu -> 0 <= mu ->
  forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 -> 0 <= Tq k0 k1 k2 k3.
Proof.
  intros H Hm k0 k1 k2 k3 Hk. specialize (H k0 k1 k2 k3).
  rewrite pencil in H. nra.
Qed.

Theorem strong_condition mu : psd_mu mu -> 0 <= mu + trace / 2 ->
  forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 ->
  0 <= Tq k0 k1 k2 k3 - trace / 2 * eta k0 k1 k2 k3.
Proof.
  intros H Hm k0 k1 k2 k3 Hk. specialize (H k0 k1 k2 k3).
  rewrite pencil in H. nra.
Qed.

Definition D00 (lam : R) : R := - lam + E*E - a*a - d*d.
Definition D01 : R := E*a - a*Sr - d*x.
Definition D02 : R := E*d - a*x - d*Th.
Definition D11 (lam : R) : R := lam + a*a - Sr*Sr - x*x.
Definition D12 : R := a*d - x*(Sr + Th).
Definition D22 (lam : R) : R := lam + d*d - x*x - Th*Th.
Definition D33 (lam : R) : R := lam - Ph*Ph.

Definition psd_lam (lam : R) : Prop :=
  forall k0 k1 k2 k3, 0 <= Qf (D00 lam) D01 D02 (D11 lam) D12 (D22 lam) (D33 lam) k0 k1 k2 k3.

Lemma flux_norm lam k0 k1 k2 k3 :
  w0 k0 k1 k2 k3 * w0 k0 k1 k2 k3
    - (w1 k0 k1 k2 k3 * w1 k0 k1 k2 k3 + w2 k0 k1 k2 k3 * w2 k0 k1 k2 k3 + w3 k0 k1 k2 k3 * w3 k0 k1 k2 k3)
  = Qf (D00 lam) D01 D02 (D11 lam) D12 (D22 lam) (D33 lam) k0 k1 k2 k3 - lam * eta k0 k1 k2 k3.
Proof. unfold w0, w1, w2, w3, Qf, D00, D01, D02, D11, D12, D22, D33, eta. ring. Qed.

(* the dominant condition: the flux is causal and future-directed for every future causal k *)
Theorem dominant_condition mu lam : psd_mu mu -> 0 <= mu -> psd_lam lam -> 0 <= lam ->
  forall k0 k1 k2 k3, 0 <= k0 -> eta k0 k1 k2 k3 <= 0 ->
  0 <= w0 k0 k1 k2 k3 /\
  w1 k0 k1 k2 k3 * w1 k0 k1 k2 k3 + w2 k0 k1 k2 k3 * w2 k0 k1 k2 k3 + w3 k0 k1 k2 k3 * w3 k0 k1 k2 k3
    <= w0 k0 k1 k2 k3 * w0 k0 k1 k2 k3.
Proof.
  intros Pm Hm Pl Hl k0 k1 k2 k3 Hk0 Hk. split.
  - assert (0 <= E) as HE.
    { assert (eta 1 0 0 0 <= 0) as He by (unfold eta; lra).
      pose proof (weak_condition mu Pm Hm 1 0 0 0 He) as W. unfold Tq, Qf in W. lra. }
    assert (a*a + d*d <= E*E) as HF.
    { specialize (Pl 1 0 0 0). unfold Qf, D00, D01, D02, D11, D12, D22, D33 in Pl. nra. }
    unfold eta in Hk.
    pose proof (Rle_0_sqr (a*k2 - d*k1)) as S. pose proof (Rle_0_sqr k1) as S1.
    pose proof (Rle_0_sqr k2) as S2. pose proof (Rle_0_sqr k3) as S3. pose proof (Rle_0_sqr E) as SE.
    unfold Rsqr in *.
    assert ((a*a + d*d) * (k1*k1 + k2*k2) <= (E*E) * (k1*k1 + k2*k2)) as H1
      by (apply Rmult_le_compat_r; lra).
    assert ((E*E) * (k1*k1 + k2*k2) <= (E*E) * (k0*k0)) as H2 by (apply Rmult_le_compat_l; lra).
    assert ((a*k1 + d*k2)*(a*k1 + d*k2) <= (E*k0)*(E*k0)) as HC by nra.
    assert (0 <= E * k0) as HP by (apply Rmult_le_pos; lra).
    unfold w0.
    destruct (Rle_or_lt (- (E*k0)) (a*k1 + d*k2)) as [Hge | Hlt]; [lra |].
    exfalso. nra.
  - pose proof (flux_norm lam k0 k1 k2 k3) as F. specialize (Pl k0 k1 k2 k3). nra.
Qed.

(* the null, weak, strong and dominant conditions, for every observer *)
Definition energy_conditions : Prop :=
  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 = 0 -> 0 <= Tq k0 k1 k2 k3) /\
  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 -> 0 <= Tq k0 k1 k2 k3) /\
  (forall k0 k1 k2 k3, eta k0 k1 k2 k3 <= 0 -> 0 <= Tq k0 k1 k2 k3 - trace / 2 * eta k0 k1 k2 k3) /\
  (forall k0 k1 k2 k3, 0 <= k0 -> eta k0 k1 k2 k3 <= 0 ->
     0 <= w0 k0 k1 k2 k3 /\
     w1 k0 k1 k2 k3 * w1 k0 k1 k2 k3 + w2 k0 k1 k2 k3 * w2 k0 k1 k2 k3 + w3 k0 k1 k2 k3 * w3 k0 k1 k2 k3
       <= w0 k0 k1 k2 k3 * w0 k0 k1 k2 k3).

(* the four conditions from the two pencils, positive semidefinite *)
Theorem pencil_conditions mu lam :
  psd_mu mu -> 0 <= mu -> 0 <= mu + trace / 2 -> psd_lam lam -> 0 <= lam -> energy_conditions.
Proof.
  intros Pm Hm Hs Pl Hl. unfold energy_conditions. repeat split.
  - apply (null_condition mu Pm).
  - apply (weak_condition mu Pm Hm).
  - apply (strong_condition mu Pm Hs).
  - intros. apply (dominant_condition mu lam Pm Hm Pl Hl); assumption.
  - intros. apply (dominant_condition mu lam Pm Hm Pl Hl); assumption.
Qed.

(* the four conditions from the two pencils, diagonally dominant *)
Theorem certified mu lam :
  Rabs a + Rabs d <= E - mu -> Rabs a + Rabs x <= Sr + mu -> Rabs d + Rabs x <= Th + mu -> 0 <= Ph + mu ->
  0 <= mu -> 0 <= mu + trace / 2 ->
  Rabs D01 + Rabs D02 <= D00 lam -> Rabs D01 + Rabs D12 <= D11 lam -> Rabs D02 + Rabs D12 <= D22 lam ->
  0 <= D33 lam -> 0 <= lam -> energy_conditions.
Proof.
  intros N0 N1 N2 N3 Hm Hs L0 L1 L2 L3 Hl.
  apply (pencil_conditions mu lam); try assumption; unfold psd_mu, psd_lam; apply dominant_nonneg; assumption.
Qed.

(* the four conditions from Sylvester's criterion on both pencils *)
Theorem certified_sylvester mu lam :
  0 <= mu -> 0 < E - mu -> 0 < minor2 (E - mu) a (Sr + mu) ->
  0 <= det3 (E - mu) a d (Sr + mu) x (Th + mu) -> 0 <= Ph + mu -> 0 <= mu + trace / 2 ->
  0 <= lam -> 0 < D00 lam -> 0 < minor2 (D00 lam) D01 (D11 lam) ->
  0 <= det3 (D00 lam) D01 D02 (D11 lam) D12 (D22 lam) -> 0 <= D33 lam -> energy_conditions.
Proof.
  intros Hm S0 S1 S2 S3 Hs Hl L0 L1 L2 L3.
  apply (pencil_conditions mu lam); try assumption; unfold psd_mu, psd_lam; apply sylvester_nonneg; assumption.
Qed.

End Certificates.

(* 6. Written by nacelle shells certify, which replaces everything from this line to the end of the file.
   G_* are the components of 8 pi T in terms of the jets of the profile functions, as derived by nacelle
   shells derive and compared with automatic differentiation of the metric by nacelle shells selftest. *)

(* Interval's bisection on the tree of an expression e, over the wall 1 <= r <= 2, -1 <= c <= 1 cut in r and in c
   to a depth of 40, for the goal 0 <= e.  The rows of the certificates below are proved with it: nacelle shells
   certify writes each row's tree, which Interval's tactic would build by reification, and the kernel runs the
   bisection once, at Qed, on its virtual machine. *)
Definition wall_hyps : list (list Reify.hyp) :=
  ((Reify.Hlele true true (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2)) :: nil)
   :: (Reify.Hlele true true (Tree.Econst (Tree.Int (-1))) (Tree.Econst (Tree.Int 1)) :: nil) :: nil)%list.

Definition wall_goal : Reify.gol := Reify.Gge true (Tree.Econst (Tree.Int 0)).

Definition wall_check (e : Tree.expr) : bool :=
  match Prog.extract e 2 with
  | Prog.Eabort => false
  | Prog.Eprog lp lc => Private.IT2.IH.eval_bisect 53%Z 40 (0%nat :: 0%nat :: 1%nat :: nil)%list wall_hyps lp lc wall_goal
  end.

Lemma wall_nonneg e : wall_check e = true ->
  forall r c, 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= Tree.eval e (r :: c :: nil).
Proof.
  intros H r c. generalize (Prog.extract_correct e (r :: c :: nil)%list).
  change (length (r :: c :: nil)%list) with 2%nat. unfold wall_check in H.
  destruct (Prog.extract e 2) as [|lp lc]; [discriminate |].
  intros <-. exact (Private.IT2.IH.eval_bisect_correct 53%Z 40 (0%nat :: 0%nat :: 1%nat :: nil)%list (r :: c :: nil)%list
                      wall_hyps lp lc wall_goal H).
Qed.

Ltac wall e r c :=
  let H := fresh in
  assert (H : wall_check e = true) by vm_cast_no_check (eq_refl true);
  exact_no_check (wall_nonneg e H r c).

(* field's normalisation of the two sides of an identity in r, c and s, written out as field's expressions over the
   atoms r - 1, r, c and s, with r replaced by (r - 1) + 1, so that the normal forms are polynomials in the depth
   r - 1.  The closed forms of the certificates below are proved with it, the kernel comparing the normal forms
   once, at Qed, and the denominators shown nonzero by Interval. *)
Definition depth_subst : list (Ring_polynom.PExpr BinNums.Z * Ring_polynom.PExpr BinNums.Z) :=
  ((Ring_polynom.PEX _ 2, Ring_polynom.PEadd (Ring_polynom.PEX _ 1) (Ring_polynom.PEc 1%Z)) :: nil)%list.

Definition field_check (fe1 fe2 : FExpr BinNums.Z) : bool :=
  let lmp := Ring_polynom.mk_monpol_list 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem depth_subst in
  let n1 := Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe1 in
  let n2 := Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe2 in
  Ring_polynom.Peq Z.eqb
    (Ring_polynom.norm_subst 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem ring_subst_niter lmp
       (Ring_polynom.PEmul (num n1) (denum n2)))
    (Ring_polynom.norm_subst 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb Z.quotrem ring_subst_niter lmp
       (Ring_polynom.PEmul (num n2) (denum n1))).

Definition field_conds (l : list R) (fe1 fe2 : FExpr BinNums.Z) : Prop :=
  PCond 0 1 Rplus Rmult Rminus Ropp eq IZR BinNat.N.to_nat pow l
    (condition (Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe1)
     ++ condition (Fnorm 0%Z 1%Z Z.add Z.mul Z.sub Z.opp Z.eqb fe2)).

Lemma field_identity r c s fe1 fe2 :
  field_check fe1 fe2 = true -> field_conds (r - 1 :: r :: c :: s :: nil)%list fe1 fe2 ->
  FEeval 0 1 Rplus Rmult Rminus Ropp Rdiv RinvImpl.Rinv IZR BinNat.N.to_nat pow (r - 1 :: r :: c :: s :: nil)%list fe1 =
  FEeval 0 1 Rplus Rmult Rminus Ropp Rdiv RinvImpl.Rinv IZR BinNat.N.to_nat pow (r - 1 :: r :: c :: s :: nil)%list fe2.
Proof.
  intros H C.
  refine (RField_field_lemma1 ring_subst_niter _ depth_subst fe1 fe2 _ _ eq_refl _ eq_refl _ eq_refl H C).
  cbn. ring.
Qed.

Ltac nonzero := repeat split; first [ exact I | lra | apply Rgt_not_eq; interval | apply Rlt_not_eq; interval ].

Ltac identity r c s fe1 fe2 :=
  let N := fresh in
  let C := fresh in
  assert (N : field_check fe1 fe2 = true) by vm_cast_no_check (eq_refl true);
  assert (C : field_conds (r - 1 :: r :: c :: s :: nil)%list fe1 fe2)
    by (unfold field_conds; apply RField_lemma5; intros lock lock_def; vm_compute; rewrite lock_def;
        clear lock lock_def; cbn [PCond Ring_polynom.PEeval BinList.nth BinList.jump List.tl List.hd
                                  BinNat.N.to_nat BinPos.Pos.to_nat BinPos.Pos.iter_op Nat.add]; nonzero);
  exact_no_check (field_identity r c s fe1 fe2 N C).

(* At the inner face of a drag shell matched C^1 to its flat cavity, the energy density E, the radial
   momentum T0r and the tangential momentum T0th seen by the free-fall observer are rho0, -D cos(th) rho0
   and D sin(th) h_rr / (16 pi psi^4), where D = psi^2 b / N is the cavity drift.  The face theorem: where
   rho0 > 0, a causal energy flux for that observer requires D^2 cos(th)^2 <= 1. *)
Module Face.

Definition G_00 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr h h_r h_rr b : R) : R :=
  (((-1) * (b ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-1) * (b ^ 2) * (h ^ 2) * (psi ^ 5)) + ((-32) * psi_r * r * (N ^ 2)) + ((-16) * psi_rr * (N ^ 2) * (r ^ 2)) + ((-1) * (b ^ 2) * (h_r ^ 2) * (psi ^ 5) * (r ^ 2)) + (2 * f * h * (b ^ 2) * (psi ^ 5)) + (5 * (b ^ 2) * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + (5 * (b ^ 2) * (c ^ 2) * (h ^ 2) * (psi ^ 5)) + ((b ^ 2) * (c ^ 2) * (h_r ^ 2) * (psi ^ 5) * (r ^ 2)) + ((-10) * f * h * (b ^ 2) * (c ^ 2) * (psi ^ 5)) + ((-2) * f * h_r * r * (b ^ 2) * (psi ^ 5)) + (2 * h * h_r * r * (b ^ 2) * (psi ^ 5)) + ((-8) * f_r * h * r * (b ^ 2) * (c ^ 2) * (psi ^ 5)) + ((-2) * h * h_r * r * (b ^ 2) * (c ^ 2) * (psi ^ 5)) + (2 * f * h_r * r * (b ^ 2) * (c ^ 2) * (psi ^ 5)) + (8 * f * f_r * r * (b ^ 2) * (c ^ 2) * (psi ^ 5)) + (32 * psi_r * r * (b ^ 2) * (c ^ 2) * (f ^ 2) * (psi ^ 4)) + (48 * (b ^ 2) * (c ^ 2) * (f ^ 2) * (psi ^ 3) * (psi_r ^ 2) * (r ^ 2)) + ((-32) * f * h * psi_r * r * (b ^ 2) * (c ^ 2) * (psi ^ 4)) + (16 * f * f_r * psi_r * (b ^ 2) * (c ^ 2) * (psi ^ 4) * (r ^ 2))) / (4 * (N ^ 2) * (psi ^ 5) * (r ^ 2)).

Definition G_01 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr h h_r h_rr b : R) : R :=
  ((N * b * c * f * (psi ^ 2)) + ((-1) * N * b * c * h * (psi ^ 2)) + ((-1) * N * b * c * h_r * r * (psi ^ 2)) + ((-4) * N * b * c * f * (psi_r ^ 2) * (r ^ 2)) + ((-2) * N_r * b * c * f * r * (psi ^ 2)) + (2 * N_r * b * c * h * r * (psi ^ 2)) + ((-4) * N * b * c * h * psi * psi_r * r) + ((-4) * N_r * b * c * f * psi * psi_r * (r ^ 2)) + (4 * N * b * c * f * psi * psi_r * r) + (4 * N * b * c * f * psi * psi_rr * (r ^ 2))) / ((N ^ 2) * (psi ^ 4) * (r ^ 2)).

Definition G_02 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr h h_r h_rr b : R) : R :=
  ((N_r * b * h * psi * s) + ((-1) * N * b * f_r * psi * s) + ((-1) * N_r * b * f * psi * s) + ((-6) * N * b * h * psi_r * s) + ((-2) * N * b * f * psi_r * s) + (2 * N * b * h_r * psi * s) + (N * b * h_rr * psi * r * s) + ((-1) * N_r * b * h_r * psi * r * s) + (6 * N * b * h_r * psi_r * r * s)) / (2 * r * (N ^ 2) * (psi ^ 3)).

Section At.
Variables r c s N psi rho0 D Nrr Frr Hrr : R.
Hypotheses (Hr : 0 < r) (HN : 0 < N) (Hpsi : 0 < psi).

(* the jets at the face: f = h = 1 and every first derivative zero, the density rho0 fixing psi_rr,
   and the shift amplitude b = D N / psi^2 *)
Definition jets (G : R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R) : R :=
  G r c s N 0 Nrr psi 0 (- (2 * PI * psi^5 * rho0)) 1 0 Frr 1 0 Hrr (D * N / psi^2).

Ltac nz := repeat split; try apply PI_neq0; try lra; try (apply pow_nonzero; lra);
           try (apply Rgt_not_eq; apply pow_lt; lra).

Lemma face_00 : jets G_00 = 8 * PI * (rho0).
Proof.
  unfold jets, G_00. field. nz.
Qed.

Lemma face_01 : jets G_01 = 8 * PI * (- (D * c * rho0)).
Proof.
  unfold jets, G_01. field. nz.
Qed.

Lemma face_02 : jets G_02 = 8 * PI * (D * s * Hrr / (16 * PI * psi^4)).
Proof.
  unfold jets, G_02. field. nz.
Qed.

(* the energy flux the free-fall observer sees is (E, -T0r, -T0th, 0); the dominant condition makes it causal *)
Theorem face_drift :
  0 < rho0 ->
  (jets G_01 / (8 * PI))^2 + (jets G_02 / (8 * PI))^2 <= (jets G_00 / (8 * PI))^2 ->
  D^2 * c^2 <= 1.
Proof.
  intros Hrho Hflux.
  rewrite face_00, face_01, face_02 in Hflux.
  replace (8 * PI * rho0 / (8 * PI)) with rho0 in Hflux by (field; apply PI_neq0).
  replace (8 * PI * - (D * c * rho0) / (8 * PI)) with (- (D * c * rho0)) in Hflux
    by (field; apply PI_neq0).
  set (t := 8 * PI * (D * s * Hrr / (16 * PI * psi ^ 4)) / (8 * PI)) in Hflux.
  assert (0 <= t^2) by (apply pow2_ge_0).
  assert ((D * c * rho0)^2 <= rho0^2) as Hb by nra.
  assert (0 < rho0^2) by (apply pow_lt; lra).
  replace ((D * c * rho0)^2) with ((D^2 * c^2) * rho0^2) in Hb by ring.
  nra.
Qed.

End At.
End Face.

(* Drag shell on 1 <= r <= 2 with M = 3/2 and shift amplitude 1/100.

   G_* are the components of 8 pi T for the drag family in terms of the jets of its profile
   functions.  Proved here: the components on this shell, the null, weak, strong and dominant energy
   conditions for every observer at every point of the wall, and that the radial jets are derivatives
   of one another.  On this shell h = f, and G_* take the jets of f for those of h. *)
Module Drag.

Definition G_00 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  (((-32) * psi_r * (N ^ 2)) + ((-1) * r * (f_r ^ 2) * (psi ^ 5)) + ((-16) * psi_rr * r * (N ^ 2)) + (r * (c ^ 2) * (f_r ^ 2) * (psi ^ 5)) + (48 * r * (c ^ 2) * (f ^ 2) * (psi ^ 3) * (psi_r ^ 2)) + (16 * f * f_r * psi_r * r * (c ^ 2) * (psi ^ 4))) / (4 * r * (N ^ 2) * (psi ^ 5)).

Definition G_01 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  (((-1) * N * c * f_r * (psi ^ 2)) + ((-4) * N * c * f * r * (psi_r ^ 2)) + ((-4) * N_r * c * f * psi * psi_r * r) + (4 * N * c * f * psi * psi_rr * r)) / (r * (N ^ 2) * (psi ^ 4)).

Definition G_02 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  ((N * f_r * psi * s) + ((-8) * N * f * psi_r * s) + (N * f_rr * psi * r * s) + ((-1) * N_r * f_r * psi * r * s) + (6 * N * f_r * psi_r * r * s)) / (2 * r * (N ^ 2) * (psi ^ 3)).

Definition G_11 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  ((8 * N_r * (N ^ 2) * (psi ^ 2)) + (16 * psi * psi_r * (N ^ 3)) + (16 * r * (N ^ 3) * (psi_r ^ 2)) + (N * r * (f_r ^ 2) * (psi ^ 6)) + ((-16) * N * psi_r * (f ^ 2) * (psi ^ 5)) + ((-4) * N * f * f_r * (psi ^ 6)) + ((-1) * N * r * (c ^ 2) * (f_r ^ 2) * (psi ^ 6)) + (4 * N * f * f_r * (c ^ 2) * (psi ^ 6)) + (16 * N * psi_r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + (16 * N_r * psi * psi_r * r * (N ^ 2)) + ((-32) * N * r * (c ^ 2) * (f ^ 2) * (psi ^ 4) * (psi_r ^ 2)) + ((-16) * N * psi_rr * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + (16 * N_r * psi_r * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-16) * N * f * f_r * psi_r * r * (c ^ 2) * (psi ^ 5))) / (4 * r * (N ^ 3) * (psi ^ 6)).

Definition G_12 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  (((-1) * N * c * f * f_r * psi * s) + (N_r * c * f * f_r * psi * r * s) + ((-1) * N * c * f * f_rr * psi * r * s) + ((-6) * N * c * f * f_r * psi_r * r * s)) / (2 * psi * r * (N ^ 3)).

Definition G_22 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  (((-8) * r * (N ^ 3) * (psi_r ^ 2)) + (4 * N_r * (N ^ 2) * (psi ^ 2)) + (8 * psi * psi_r * (N ^ 3)) + ((-16) * N * psi_r * (f ^ 2) * (psi ^ 5)) + ((-3) * N * r * (f_r ^ 2) * (psi ^ 6)) + (4 * N_rr * r * (N ^ 2) * (psi ^ 2)) + (8 * psi * psi_rr * r * (N ^ 3)) + ((-1) * N * r * (c ^ 2) * (f_r ^ 2) * (psi ^ 6)) + (16 * N * psi_r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-32) * N * r * (c ^ 2) * (f ^ 2) * (psi ^ 4) * (psi_r ^ 2)) + ((-16) * N * psi_rr * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-4) * N * f * f_rr * r * (c ^ 2) * (psi ^ 6)) + (4 * N_r * f * f_r * r * (c ^ 2) * (psi ^ 6)) + (16 * N_r * psi_r * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-40) * N * f * f_r * psi_r * r * (c ^ 2) * (psi ^ 5))) / (4 * r * (N ^ 3) * (psi ^ 6)).

Definition G_33 (r c s N N_r N_rr psi psi_r psi_rr f f_r f_rr : R) : R :=
  (((-8) * r * (N ^ 3) * (psi_r ^ 2)) + (4 * N_r * (N ^ 2) * (psi ^ 2)) + (8 * psi * psi_r * (N ^ 3)) + ((-1) * N * r * (f_r ^ 2) * (psi ^ 6)) + ((-16) * N * psi_r * (f ^ 2) * (psi ^ 5)) + ((-4) * N * f * f_r * (psi ^ 6)) + (4 * N_rr * r * (N ^ 2) * (psi ^ 2)) + (8 * psi * psi_rr * r * (N ^ 3)) + ((-3) * N * r * (c ^ 2) * (f_r ^ 2) * (psi ^ 6)) + (4 * N * f * f_r * (c ^ 2) * (psi ^ 6)) + (16 * N * psi_r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-32) * N * r * (c ^ 2) * (f ^ 2) * (psi ^ 4) * (psi_r ^ 2)) + ((-16) * N * psi_rr * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-4) * N * f * f_rr * r * (c ^ 2) * (psi ^ 6)) + (4 * N_r * f * f_r * r * (c ^ 2) * (psi ^ 6)) + (16 * N_r * psi_r * r * (c ^ 2) * (f ^ 2) * (psi ^ 5)) + ((-40) * N * f * f_r * psi_r * r * (c ^ 2) * (psi ^ 5))) / (4 * r * (N ^ 3) * (psi ^ 6)).

Definition j_N (r : R) : R := (17 + (3 * ((r - 1) ^ 2))) / (47 + ((-3) * ((r - 1) ^ 2))).
Definition j_N_r (r : R) : R := (384 * (r - 1)) / ((47 + ((-3) * ((r - 1) ^ 2))) ^ 2).
Definition j_N_rr (r : R) : R := (18048 + (3456 * ((r - 1) ^ 2))) / ((47 + ((-3) * ((r - 1) ^ 2))) ^ 3).
Definition j_psi (r : R) : R := (47 + ((-3) * ((r - 1) ^ 2))) / 32.
Definition j_psi_r (r : R) : R := ((-3) * (r - 1)) / 16.
Definition j_psi_rr (r : R) : R := (-3) / 16.
Definition j_f (r : R) : R := (1 + (((r - 1) ^ 2) * ((-3) + (2 * (r - 1))))) / 100.
Definition j_f_r (r : R) : R := ((r - 1) * ((-3) + (3 * (r - 1)))) / 50.
Definition j_f_rr (r : R) : R := ((-3) + (6 * (r - 1))) / 50.

Definition x_00 (r c s : R) : R := (((-72729231360000) + ((r - 1) * ((-218187694080000) + ((r - 1) * ((-21109532395833) + ((-6210892134567) * (c ^ 2)) + ((r - 1) * ((-81567029524167) + (4460531041143 * (c ^ 2)) + ((r - 1) * ((-8861803941348) + (14026252660020 * (c ^ 2)) + ((r - 1) * ((-197892698652) + ((-9471519377172) * (c ^ 2)) + ((r - 1) * (2427387554088 + ((-11038478802264) * (c ^ 2)) + ((r - 1) * ((-2427387554088) + (6950679910200 * (c ^ 2)) + ((r - 1) * ((-431617543812) + (3830679748116 * (c ^ 2)) + ((r - 1) * (431617543812 + ((-2293102293876) * (c ^ 2)) + ((r - 1) * (44150730750 + ((-668406449070) * (c ^ 2)) + ((r - 1) * ((-44150730750) + (388086988590 * (c ^ 2)) + ((r - 1) * ((-2750496588) + (64218240828 * (c ^ 2)) + ((r - 1) * (2750496588 + ((-36572997852) * (c ^ 2)) + ((r - 1) * (103611312 + ((-3471439680) * (c ^ 2)) + ((r - 1) * ((-103611312) + (1951410528 * (c ^ 2)) + ((r - 1) * ((-2178252) + (99351036 * (c ^ 2)) + ((r - 1) * (2178252 + ((-55331100) * (c ^ 2)) + ((r - 1) * (19683 + ((-1174419) * (c ^ 2)) + ((r - 1) * ((-19683) + (649539 * (c ^ 2))))))))))))))))))))))))))))))))))))))))) / (10000 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 5) * ((17 + (3 * ((r - 1) ^ 2))) ^ 2) * r)).

Definition x_01 (r c s : R) : R := ((((-104448) * c) + ((r - 1) * ((1122816 * c) + ((r - 1) * (((-895488) * c) + ((r - 1) * ((261120 * c) + ((r - 1) * (((-402432) * c) + ((r - 1) * (((-32256) * c) + (50688 * c * (r - 1))))))))))))) / (25 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 2) * ((17 + (3 * ((r - 1) ^ 2))) ^ 2) * r)).

Definition x_02 (r c s : R) : R := s * (((-613632) + ((r - 1) * (104448 + ((r - 1) * (2536704 + ((r - 1) * ((-294912) + ((r - 1) * ((-258816) + ((r - 1) * ((-55296) + ((-66816) * (r - 1))))))))))))) / (25 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 2) * ((17 + (3 * ((r - 1) ^ 2))) ^ 2) * r)).

Definition x_11 (r c s : R) : R := ((((-4397919854232) * (c ^ 2)) + ((r - 1) * ((-1147011948541458) + (51675558287226 * (c ^ 2)) + ((r - 1) * (410537609016387 + (82757540360901 * (c ^ 2)) + ((r - 1) * (543331575752013 + ((-260865038037045) * (c ^ 2)) + ((r - 1) * ((-37456359009027) + ((-2375726371341) * (c ^ 2)) + ((r - 1) * (114672969859485 + (139248465452547 * (c ^ 2)) + ((r - 1) * (118718544028716 + (89606492938356 * (c ^ 2)) + ((r - 1) * (5036840691252 + ((-76695180940116) * (c ^ 2)) + ((r - 1) * ((-12153274942428) + ((-47265826507956) * (c ^ 2)) + ((r - 1) * (5660050877928 + (23809236371784 * (c ^ 2)) + ((r - 1) * ((-1497811438278) + (7385977184310 * (c ^ 2)) + ((r - 1) * ((-212610191418) + ((-2514297554550) * (c ^ 2)) + ((r - 1) * (489799886742 + ((-292466976534) * (c ^ 2)) + ((r - 1) * ((-78398876322) + ((-82260828990) * (c ^ 2)) + ((r - 1) * ((-50390045892) + ((-32799569436) * (c ^ 2)) + ((r - 1) * (11518000740 + (37357090812 * (c ^ 2)) + ((r - 1) * (2624461236 + (3989149236 * (c ^ 2)) + ((r - 1) * ((-689854158) + ((-3049548426) * (c ^ 2)) + ((r - 1) * ((-70156773) + ((-159198291) * (c ^ 2)) + ((r - 1) * (19943253 + (107725059 * (c ^ 2)) + ((r - 1) * (767637 + (2276667 * (c ^ 2)) + ((r - 1) * ((-229635) + ((-1449981) * (c ^ 2))))))))))))))))))))))))))))))))))))))))))))) / (10000 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 5) * ((17 + (3 * ((r - 1) ^ 2))) ^ 3) * r)).

Definition x_12 (r c s : R) : R := s * (((112659 * c) + (((r - 1) ^ 2) * (((-810891) * c) + ((r - 1) * ((225318 * c) + ((r - 1) * ((1534338 * c) + ((r - 1) * (((-945828) * c) + ((r - 1) * (((-333234) * c) + ((r - 1) * ((231192 * c) + ((r - 1) * (((-41877) * c) + ((r - 1) * ((27108 * c) + ((r - 1) * ((3645 * c) + ((-2430) * c * (r - 1))))))))))))))))))))) / (10000 * ((17 + (3 * ((r - 1) ^ 2))) ^ 3) * r)).

Definition x_22 (r c s : R) : R := (((-545469235200000) + ((-56073478141458) * (c ^ 2)) + ((r - 1) * ((-1095336390254232) + (51675558287226 * (c ^ 2)) + ((r - 1) * ((-69072388347483) + (471197343956745 * (c ^ 2)) + ((r - 1) * ((-590281394139405) + ((-364216154611497) * (c ^ 2)) + ((r - 1) * ((-368486717428173) + ((-826973314409463) * (c ^ 2)) + ((r - 1) * (216428475936429 + (606074722920879 * (c ^ 2)) + ((r - 1) * (117435324148020 + (482358431596452 * (c ^ 2)) + ((r - 1) * ((-74037976607316) + ((-325411584611364) * (c ^ 2)) + ((r - 1) * ((-10950648281892) + ((-106485025060008) * (c ^ 2)) + ((r - 1) * (4205235243348 + (63163902674232 * (c ^ 2)) + ((r - 1) * ((-2072595085002) + (7672359490254 * (c ^ 2)) + ((r - 1) * (1507883039514 + ((-2888695751310) * (c ^ 2)) + ((r - 1) * (563522213322 + (567421529670 * (c ^ 2)) + ((r - 1) * ((-327835435050) + ((-632690807382) * (c ^ 2)) + ((r - 1) * ((-55430544348) + ((-137439331308) * (c ^ 2)) + ((r - 1) * (29721229884 + (105844168044 * (c ^ 2)) + ((r - 1) * (2823373260 + (9769429602 * (c ^ 2)) + ((r - 1) * ((-1440506916) + ((-6867840474) * (c ^ 2)) + ((r - 1) * ((-74447667) + ((-318315663) * (c ^ 2)) + ((r - 1) * (36673803 + (213409647 * (c ^ 2)) + ((r - 1) * (807003 + (4048137 * (c ^ 2)) + ((r - 1) * ((-387099) + ((-2630961) * (c ^ 2))))))))))))))))))))))))))))))))))))))))))))) / (10000 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 5) * ((17 + (3 * ((r - 1) ^ 2))) ^ 3) * r)).

Definition x_33 (r c s : R) : R := (((-545469235200000) + ((-56073478141458) * (c ^ 2)) + ((r - 1) * ((-1147011948541458) + (103351116574452 * (c ^ 2)) + ((r - 1) * ((-172423504921935) + (574548460531197 * (c ^ 2)) + ((r - 1) * ((-266258181349665) + ((-688239367401237) * (c ^ 2)) + ((r - 1) * ((-443898107869857) + ((-751561923967779) * (c ^ 2)) + ((r - 1) * (80586969600315 + (741916229256993 * (c ^ 2)) + ((r - 1) * (144681510979620 + (455112244764852 * (c ^ 2)) + ((r - 1) * ((-58297374899652) + ((-341152186319028) * (c ^ 2)) + ((r - 1) * ((-12263946107508) + ((-105171727234392) * (c ^ 2)) + ((r - 1) * (5770722043008 + (61598415874572 * (c ^ 2)) + ((r - 1) * ((-2586391855650) + (8186156260902 * (c ^ 2)) + ((r - 1) * (875970225954 + ((-2256782937750) * (c ^ 2)) + ((r - 1) * (661187387250 + (469756355742 * (c ^ 2)) + ((r - 1) * ((-249786376830) + ((-710739865602) * (c ^ 2)) + ((r - 1) * ((-63370240812) + ((-129499634844) * (c ^ 2)) + ((r - 1) * (24498195660 + (111067202268 * (c ^ 2)) + ((r - 1) * (3172068540 + (9420734322 * (c ^ 2)) + ((r - 1) * ((-1237461462) + ((-7070885928) * (c ^ 2)) + ((r - 1) * ((-82557063) + ((-310206267) * (c ^ 2)) + ((r - 1) * (32343543 + (217739907 * (c ^ 2)) + ((r - 1) * (885735 + (3969405 * (c ^ 2)) + ((r - 1) * ((-347733) + ((-2670327) * (c ^ 2))))))))))))))))))))))))))))))))))))))))))))) / (10000 * (((-47) + (3 * ((r - 1) ^ 2))) ^ 5) * ((17 + (3 * ((r - 1) ^ 2))) ^ 3) * r)).

(* field's expressions of the jets, of which those of the identities below are built *)
Definition fe_N : FExpr BinNums.Z := (FEdiv (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (FEadd (FEc 47%Z) (FEmul (FEc (-3)%Z) (FEpow (FEX _ 1) (Npos 2))))).
Definition fe_N_r : FExpr BinNums.Z := (FEdiv (FEmul (FEc 384%Z) (FEX _ 1)) (FEpow (FEadd (FEc 47%Z) (FEmul (FEc (-3)%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))).
Definition fe_N_rr : FExpr BinNums.Z := (FEdiv (FEadd (FEc 18048%Z) (FEmul (FEc 3456%Z) (FEpow (FEX _ 1) (Npos 2)))) (FEpow (FEadd (FEc 47%Z) (FEmul (FEc (-3)%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 3))).
Definition fe_psi : FExpr BinNums.Z := (FEdiv (FEadd (FEc 47%Z) (FEmul (FEc (-3)%Z) (FEpow (FEX _ 1) (Npos 2)))) (FEc 32%Z)).
Definition fe_psi_r : FExpr BinNums.Z := (FEdiv (FEmul (FEc (-3)%Z) (FEX _ 1)) (FEc 16%Z)).
Definition fe_psi_rr : FExpr BinNums.Z := (FEdiv (FEc (-3)%Z) (FEc 16%Z)).
Definition fe_f : FExpr BinNums.Z := (FEdiv (FEadd (FEc 1%Z) (FEmul (FEpow (FEX _ 1) (Npos 2)) (FEadd (FEc (-3)%Z) (FEmul (FEc 2%Z) (FEX _ 1))))) (FEc 100%Z)).
Definition fe_f_r : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc (-3)%Z) (FEmul (FEc 3%Z) (FEX _ 1)))) (FEc 50%Z)).
Definition fe_f_rr : FExpr BinNums.Z := (FEdiv (FEadd (FEc (-3)%Z) (FEmul (FEc 6%Z) (FEX _ 1))) (FEc 50%Z)).

Definition fe_G_00 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEc (-32)%Z) fe_psi_r) (FEpow fe_N (Npos 2))) (FEmul (FEmul (FEmul (FEc (-1)%Z) (FEX _ 2)) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_psi_rr) (FEX _ 2)) (FEpow fe_N (Npos 2)))) (FEmul (FEmul (FEmul (FEX _ 2) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 48%Z) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 3))) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_f) fe_f_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 4)))) (FEmul (FEmul (FEmul (FEc 4%Z) (FEX _ 2)) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 5)))).
Definition fe_x_00 : FExpr BinNums.Z := (FEdiv (FEadd (FEc (-72729231360000)%Z) (FEmul (FEX _ 1) (FEadd (FEc (-218187694080000)%Z) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-21109532395833)%Z) (FEmul (FEc (-6210892134567)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-81567029524167)%Z) (FEmul (FEc 4460531041143%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-8861803941348)%Z) (FEmul (FEc 14026252660020%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-197892698652)%Z) (FEmul (FEc (-9471519377172)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2427387554088%Z) (FEmul (FEc (-11038478802264)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2427387554088)%Z) (FEmul (FEc 6950679910200%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-431617543812)%Z) (FEmul (FEc 3830679748116%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 431617543812%Z) (FEmul (FEc (-2293102293876)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 44150730750%Z) (FEmul (FEc (-668406449070)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-44150730750)%Z) (FEmul (FEc 388086988590%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2750496588)%Z) (FEmul (FEc 64218240828%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2750496588%Z) (FEmul (FEc (-36572997852)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 103611312%Z) (FEmul (FEc (-3471439680)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-103611312)%Z) (FEmul (FEc 1951410528%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2178252)%Z) (FEmul (FEc 99351036%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2178252%Z) (FEmul (FEc (-55331100)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 19683%Z) (FEmul (FEc (-1174419)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-19683)%Z) (FEmul (FEc 649539%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))))))))))))))))))))))) (FEmul (FEmul (FEmul (FEc 10000%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 5))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))) (FEX _ 2))).
Lemma x_00_closed r c s : 1 <= r <= 2 -> G_00 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_00 r c s.
Proof. intros Hr. identity r c s fe_G_00 fe_x_00. Qed.

Definition fe_G_01 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 3)) fe_f_r) (FEpow fe_psi (Npos 2))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N) (FEX _ 3)) fe_f) (FEX _ 2)) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N_r) (FEX _ 3)) fe_f) fe_psi) fe_psi_r) (FEX _ 2))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N) (FEX _ 3)) fe_f) fe_psi) fe_psi_rr) (FEX _ 2))) (FEmul (FEmul (FEX _ 2) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 4)))).
Definition fe_x_01 : FExpr BinNums.Z := (FEdiv (FEadd (FEmul (FEc (-104448)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 1122816%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-895488)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 261120%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-402432)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-32256)%Z) (FEX _ 3)) (FEmul (FEmul (FEc 50688%Z) (FEX _ 3)) (FEX _ 1))))))))))))) (FEmul (FEmul (FEmul (FEc 25%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))) (FEX _ 2))).
Lemma x_01_closed r c s : 1 <= r <= 2 -> G_01 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_01 r c s.
Proof. intros Hr. identity r c s fe_G_01 fe_x_01. Qed.

Definition fe_G_02 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul fe_N fe_f_r) fe_psi) (FEX _ 4)) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_N) fe_f) fe_psi_r) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul fe_N fe_f_rr) fe_psi) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N_r) fe_f_r) fe_psi) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 6%Z) fe_N) fe_f_r) fe_psi_r) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEc 2%Z) (FEX _ 2)) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 3)))).
Definition fe_x_02 : FExpr BinNums.Z := (FEmul (FEX _ 4) (FEdiv (FEadd (FEc (-613632)%Z) (FEmul (FEX _ 1) (FEadd (FEc 104448%Z) (FEmul (FEX _ 1) (FEadd (FEc 2536704%Z) (FEmul (FEX _ 1) (FEadd (FEc (-294912)%Z) (FEmul (FEX _ 1) (FEadd (FEc (-258816)%Z) (FEmul (FEX _ 1) (FEadd (FEc (-55296)%Z) (FEmul (FEc (-66816)%Z) (FEX _ 1))))))))))))) (FEmul (FEmul (FEmul (FEc 25%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 2))) (FEX _ 2)))).
Lemma x_02_closed r c s : 1 <= r <= 2 -> G_02 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_02 r c s.
Proof. intros Hr. identity r c s fe_G_02 fe_x_02. Qed.

Definition fe_G_11 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc 8%Z) fe_N_r) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 2))) (FEmul (FEmul (FEmul (FEc 16%Z) fe_psi) fe_psi_r) (FEpow fe_N (Npos 3)))) (FEmul (FEmul (FEmul (FEc 16%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul fe_N (FEX _ 2)) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_r) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N) fe_f) fe_f_r) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N) fe_f) fe_f_r) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N) fe_psi_r) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N_r) fe_psi) fe_psi_r) (FEX _ 2)) (FEpow fe_N (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-32)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 4))) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_rr) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_f) fe_f_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEc 4%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi (Npos 6)))).
Definition fe_x_11 : FExpr BinNums.Z := (FEdiv (FEadd (FEmul (FEc (-4397919854232)%Z) (FEpow (FEX _ 3) (Npos 2))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1147011948541458)%Z) (FEmul (FEc 51675558287226%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 410537609016387%Z) (FEmul (FEc 82757540360901%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 543331575752013%Z) (FEmul (FEc (-260865038037045)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-37456359009027)%Z) (FEmul (FEc (-2375726371341)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 114672969859485%Z) (FEmul (FEc 139248465452547%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 118718544028716%Z) (FEmul (FEc 89606492938356%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 5036840691252%Z) (FEmul (FEc (-76695180940116)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-12153274942428)%Z) (FEmul (FEc (-47265826507956)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 5660050877928%Z) (FEmul (FEc 23809236371784%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1497811438278)%Z) (FEmul (FEc 7385977184310%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-212610191418)%Z) (FEmul (FEc (-2514297554550)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 489799886742%Z) (FEmul (FEc (-292466976534)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-78398876322)%Z) (FEmul (FEc (-82260828990)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-50390045892)%Z) (FEmul (FEc (-32799569436)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 11518000740%Z) (FEmul (FEc 37357090812%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2624461236%Z) (FEmul (FEc 3989149236%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-689854158)%Z) (FEmul (FEc (-3049548426)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-70156773)%Z) (FEmul (FEc (-159198291)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 19943253%Z) (FEmul (FEc 107725059%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 767637%Z) (FEmul (FEc 2276667%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-229635)%Z) (FEmul (FEc (-1449981)%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))))))))))))))))))))))))))) (FEmul (FEmul (FEmul (FEc 10000%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 5))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 3))) (FEX _ 2))).
Lemma x_11_closed r c s : 1 <= r <= 2 -> G_11 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_11 r c s.
Proof. intros Hr. identity r c s fe_G_11 fe_x_11. Qed.

Definition fe_G_12 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 3)) fe_f) fe_f_r) fe_psi) (FEX _ 4)) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul fe_N_r (FEX _ 3)) fe_f) fe_f_r) fe_psi) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 3)) fe_f) fe_f_rr) fe_psi) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-6)%Z) fe_N) (FEX _ 3)) fe_f) fe_f_r) fe_psi_r) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEc 2%Z) fe_psi) (FEX _ 2)) (FEpow fe_N (Npos 3)))).
Definition fe_x_12 : FExpr BinNums.Z := (FEmul (FEX _ 4) (FEdiv (FEadd (FEmul (FEc 112659%Z) (FEX _ 3)) (FEmul (FEpow (FEX _ 1) (Npos 2)) (FEadd (FEmul (FEc (-810891)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 225318%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 1534338%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-945828)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-333234)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 231192%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-41877)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 27108%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc 3645%Z) (FEX _ 3)) (FEmul (FEmul (FEc (-2430)%Z) (FEX _ 3)) (FEX _ 1))))))))))))))))))))) (FEmul (FEmul (FEc 10000%Z) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 3))) (FEX _ 2)))).
Lemma x_12_closed r c s : 1 <= r <= 2 -> G_12 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_12 r c s.
Proof. intros Hr. identity r c s fe_G_12 fe_x_12. Qed.

Definition fe_G_22 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc (-8)%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi_r (Npos 2))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_r) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 2)))) (FEmul (FEmul (FEmul (FEc 8%Z) fe_psi) fe_psi_r) (FEpow fe_N (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_r) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEc (-3)%Z) fe_N) (FEX _ 2)) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_rr) (FEX _ 2)) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_psi) fe_psi_rr) (FEX _ 2)) (FEpow fe_N (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N) fe_psi_r) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-32)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 4))) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_rr) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N) fe_f) fe_f_rr) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_r) fe_f) fe_f_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-40)%Z) fe_N) fe_f) fe_f_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEc 4%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi (Npos 6)))).
Definition fe_x_22 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEc (-545469235200000)%Z) (FEmul (FEc (-56073478141458)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1095336390254232)%Z) (FEmul (FEc 51675558287226%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-69072388347483)%Z) (FEmul (FEc 471197343956745%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-590281394139405)%Z) (FEmul (FEc (-364216154611497)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-368486717428173)%Z) (FEmul (FEc (-826973314409463)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 216428475936429%Z) (FEmul (FEc 606074722920879%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 117435324148020%Z) (FEmul (FEc 482358431596452%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-74037976607316)%Z) (FEmul (FEc (-325411584611364)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-10950648281892)%Z) (FEmul (FEc (-106485025060008)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 4205235243348%Z) (FEmul (FEc 63163902674232%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2072595085002)%Z) (FEmul (FEc 7672359490254%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 1507883039514%Z) (FEmul (FEc (-2888695751310)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 563522213322%Z) (FEmul (FEc 567421529670%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-327835435050)%Z) (FEmul (FEc (-632690807382)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-55430544348)%Z) (FEmul (FEc (-137439331308)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 29721229884%Z) (FEmul (FEc 105844168044%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2823373260%Z) (FEmul (FEc 9769429602%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1440506916)%Z) (FEmul (FEc (-6867840474)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-74447667)%Z) (FEmul (FEc (-318315663)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 36673803%Z) (FEmul (FEc 213409647%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 807003%Z) (FEmul (FEc 4048137%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-387099)%Z) (FEmul (FEc (-2630961)%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))))))))))))))))))))))))))) (FEmul (FEmul (FEmul (FEc 10000%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 5))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 3))) (FEX _ 2))).
Lemma x_22_closed r c s : 1 <= r <= 2 -> G_22 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_22 r c s.
Proof. intros Hr. identity r c s fe_G_22 fe_x_22. Qed.

Definition fe_G_33 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc (-8)%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi_r (Npos 2))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_r) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 2)))) (FEmul (FEmul (FEmul (FEc 8%Z) fe_psi) fe_psi_r) (FEpow fe_N (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_N) (FEX _ 2)) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_r) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N) fe_f) fe_f_r) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_rr) (FEX _ 2)) (FEpow fe_N (Npos 2))) (FEpow fe_psi (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_psi) fe_psi_rr) (FEX _ 2)) (FEpow fe_N (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-3)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f_r (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N) fe_f) fe_f_r) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N) fe_psi_r) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-32)%Z) fe_N) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 4))) (FEpow fe_psi_r (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-16)%Z) fe_N) fe_psi_rr) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_N) fe_f) fe_f_rr) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_N_r) fe_f) fe_f_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 6)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_N_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_f (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-40)%Z) fe_N) fe_f) fe_f_r) fe_psi_r) (FEX _ 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow fe_psi (Npos 5)))) (FEmul (FEmul (FEmul (FEc 4%Z) (FEX _ 2)) (FEpow fe_N (Npos 3))) (FEpow fe_psi (Npos 6)))).
Definition fe_x_33 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEc (-545469235200000)%Z) (FEmul (FEc (-56073478141458)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1147011948541458)%Z) (FEmul (FEc 103351116574452%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-172423504921935)%Z) (FEmul (FEc 574548460531197%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-266258181349665)%Z) (FEmul (FEc (-688239367401237)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-443898107869857)%Z) (FEmul (FEc (-751561923967779)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 80586969600315%Z) (FEmul (FEc 741916229256993%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 144681510979620%Z) (FEmul (FEc 455112244764852%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-58297374899652)%Z) (FEmul (FEc (-341152186319028)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-12263946107508)%Z) (FEmul (FEc (-105171727234392)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 5770722043008%Z) (FEmul (FEc 61598415874572%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2586391855650)%Z) (FEmul (FEc 8186156260902%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 875970225954%Z) (FEmul (FEc (-2256782937750)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 661187387250%Z) (FEmul (FEc 469756355742%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-249786376830)%Z) (FEmul (FEc (-710739865602)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-63370240812)%Z) (FEmul (FEc (-129499634844)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 24498195660%Z) (FEmul (FEc 111067202268%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 3172068540%Z) (FEmul (FEc 9420734322%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1237461462)%Z) (FEmul (FEc (-7070885928)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-82557063)%Z) (FEmul (FEc (-310206267)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 32343543%Z) (FEmul (FEc 217739907%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 885735%Z) (FEmul (FEc 3969405%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-347733)%Z) (FEmul (FEc (-2670327)%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))))))))))))))))))))))))))) (FEmul (FEmul (FEmul (FEc 10000%Z) (FEpow (FEadd (FEc (-47)%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 5))) (FEpow (FEadd (FEc 17%Z) (FEmul (FEc 3%Z) (FEpow (FEX _ 1) (Npos 2)))) (Npos 3))) (FEX _ 2))).
Lemma x_33_closed r c s : 1 <= r <= 2 -> G_33 r c s (j_N r) (j_N_r r) (j_N_rr r) (j_psi r) (j_psi_r r) (j_psi_rr r) (j_f r) (j_f_r r) (j_f_rr r) = x_33 r c s.
Proof. intros Hr. identity r c s fe_G_33 fe_x_33. Qed.

(* the components at radius r and c = cos(theta), with sin(theta) = sqrt (1 - c^2) *)
Definition tE (r c : R) : R := x_00 r c (sqrt (1 - c * c)).
Definition ta (r c : R) : R := x_01 r c (sqrt (1 - c * c)).
Definition td (r c : R) : R := x_02 r c (sqrt (1 - c * c)).
Definition tS (r c : R) : R := x_11 r c (sqrt (1 - c * c)).
Definition tx (r c : R) : R := x_12 r c (sqrt (1 - c * c)).
Definition tT (r c : R) : R := x_22 r c (sqrt (1 - c * c)).
Definition tP (r c : R) : R := x_33 r c (sqrt (1 - c * c)).

Definition rho (r : R) : R := ((-25165824) + ((-75497472) * (r - 1))) / ((((-47) + (3 * ((r - 1) ^ 2))) ^ 5) * r).
Definition mu (r : R) : R := (1 / 2) * rho r.
Definition lam (r : R) : R := (1 / 2) * (rho r * rho r).

(* the trees of rho and of the components, of which the rows' trees are built *)
Definition tree_rho : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-25165824))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-75497472))) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))) (Tree.Ebinary Tree.Mul (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))) (Tree.Evar 0))).
Definition tree_E : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-72729231360000))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-218187694080000))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-21109532395833))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-6210892134567))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-81567029524167))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 4460531041143)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-8861803941348))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 14026252660020)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-197892698652))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-9471519377172))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2427387554088)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-11038478802264))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2427387554088))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 6950679910200)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-431617543812))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3830679748116)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 431617543812)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2293102293876))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 44150730750)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-668406449070))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-44150730750))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 388086988590)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2750496588))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 64218240828)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2750496588)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-36572997852))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 103611312)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-3471439680))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-103611312))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1951410528)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2178252))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 99351036)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2178252)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-55331100))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 19683)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-1174419))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-19683))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 649539)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 10000)) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0))).
Definition tree_a : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-104448))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1122816)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-895488))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 261120)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-402432))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-32256))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 50688)) (Tree.Evar 1)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 25)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0))).
Definition tree_d : Tree.expr := (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-613632))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 104448)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2536704)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-294912))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-258816))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-55296))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-66816))) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 25)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0)))).
Definition tree_S : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-4397919854232))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1147011948541458))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 51675558287226)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 410537609016387)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 82757540360901)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 543331575752013)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-260865038037045))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-37456359009027))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2375726371341))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 114672969859485)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 139248465452547)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 118718544028716)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 89606492938356)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5036840691252)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-76695180940116))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-12153274942428))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-47265826507956))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5660050877928)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 23809236371784)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1497811438278))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 7385977184310)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-212610191418))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2514297554550))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 489799886742)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-292466976534))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-78398876322))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-82260828990))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-50390045892))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-32799569436))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 11518000740)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 37357090812)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2624461236)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3989149236)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-689854158))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-3049548426))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-70156773))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-159198291))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 19943253)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 107725059)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 767637)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2276667)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-229635))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-1449981))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 10000)) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0))).
Definition tree_x : Tree.expr := (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 112659)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-810891))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 225318)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1534338)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-945828))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-333234))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 231192)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-41877))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 27108)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3645)) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2430))) (Tree.Evar 1)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 10000)) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0)))).
Definition tree_T : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-545469235200000))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-56073478141458))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1095336390254232))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 51675558287226)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-69072388347483))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 471197343956745)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-590281394139405))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-364216154611497))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-368486717428173))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-826973314409463))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 216428475936429)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 606074722920879)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 117435324148020)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 482358431596452)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-74037976607316))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-325411584611364))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-10950648281892))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-106485025060008))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4205235243348)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 63163902674232)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2072595085002))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 7672359490254)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1507883039514)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2888695751310))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 563522213322)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 567421529670)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-327835435050))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-632690807382))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-55430544348))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-137439331308))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 29721229884)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 105844168044)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2823373260)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 9769429602)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1440506916))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-6867840474))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-74447667))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-318315663))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 36673803)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 213409647)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 807003)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 4048137)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-387099))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2630961))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 10000)) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0))).
Definition tree_P : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-545469235200000))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-56073478141458))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1147011948541458))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 103351116574452)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-172423504921935))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 574548460531197)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-266258181349665))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-688239367401237))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-443898107869857))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-751561923967779))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 80586969600315)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 741916229256993)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 144681510979620)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 455112244764852)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-58297374899652))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-341152186319028))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-12263946107508))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-105171727234392))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5770722043008)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 61598415874572)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2586391855650))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 8186156260902)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 875970225954)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2256782937750))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 661187387250)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 469756355742)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-249786376830))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-710739865602))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-63370240812))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-129499634844))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 24498195660)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 111067202268)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 3172068540)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 9420734322)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1237461462))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-7070885928))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-82557063))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-310206267))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 32343543)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 217739907)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 885735)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3969405)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-347733))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-2670327))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 10000)) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-47))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 17)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Evar 0))).

Lemma row_n0 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tE r c - mu r - Rabs (ta r c) - Rabs (td r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub tree_E (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho)) (Tree.Eunary Tree.Abs tree_a)) (Tree.Eunary Tree.Abs tree_d)) r c. Qed.

Lemma row_n1 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tS r c + mu r - Rabs (ta r c) - Rabs (tx r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add tree_S (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho)) (Tree.Eunary Tree.Abs tree_a)) (Tree.Eunary Tree.Abs tree_x)) r c. Qed.

Lemma row_n2 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tT r c + mu r - Rabs (td r c) - Rabs (tx r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add tree_T (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho)) (Tree.Eunary Tree.Abs tree_d)) (Tree.Eunary Tree.Abs tree_x)) r c. Qed.

Lemma row_n3 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tP r c + mu r.
Proof. wall (Tree.Ebinary Tree.Add tree_P (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho)) r c. Qed.

Lemma row_m r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= mu r.
Proof. wall (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho) r c. Qed.

Lemma row_s r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= mu r + trace (tE r c) (tS r c) (tT r c) (tP r c) / 2.
Proof. wall (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) tree_rho) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Eunary Tree.Neg tree_E) tree_S) tree_T) tree_P) (Tree.Econst (Tree.Int 2)))) r c. Qed.

Lemma row_d0 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D00 (tE r c) (ta r c) (td r c) (lam r) - Rabs (D01 (tE r c) (ta r c) (td r c) (tS r c) (tx r c)) - Rabs (D02 (tE r c) (ta r c) (td r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Eunary Tree.Neg (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho))) (Tree.Eunary Tree.Sqr tree_E)) (Tree.Eunary Tree.Sqr tree_a)) (Tree.Eunary Tree.Sqr tree_d)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_a) (Tree.Ebinary Tree.Mul tree_a tree_S)) (Tree.Ebinary Tree.Mul tree_d tree_x)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_d) (Tree.Ebinary Tree.Mul tree_a tree_x)) (Tree.Ebinary Tree.Mul tree_d tree_T)))) r c. Qed.

Lemma row_d1 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D11 (ta r c) (tS r c) (tx r c) (lam r) - Rabs (D01 (tE r c) (ta r c) (td r c) (tS r c) (tx r c)) - Rabs (D12 (ta r c) (td r c) (tS r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_a)) (Tree.Eunary Tree.Sqr tree_S)) (Tree.Eunary Tree.Sqr tree_x)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_a) (Tree.Ebinary Tree.Mul tree_a tree_S)) (Tree.Ebinary Tree.Mul tree_d tree_x)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_a tree_d) (Tree.Ebinary Tree.Mul tree_x (Tree.Ebinary Tree.Add tree_S tree_T))))) r c. Qed.

Lemma row_d2 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D22 (td r c) (tx r c) (tT r c) (lam r) - Rabs (D02 (tE r c) (ta r c) (td r c) (tx r c) (tT r c)) - Rabs (D12 (ta r c) (td r c) (tS r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_d)) (Tree.Eunary Tree.Sqr tree_x)) (Tree.Eunary Tree.Sqr tree_T)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_d) (Tree.Ebinary Tree.Mul tree_a tree_x)) (Tree.Ebinary Tree.Mul tree_d tree_T)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_a tree_d) (Tree.Ebinary Tree.Mul tree_x (Tree.Ebinary Tree.Add tree_S tree_T))))) r c. Qed.

Lemma row_d3 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D33 (tP r c) (lam r).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_P)) r c. Qed.

Lemma row_l r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= lam r.
Proof. wall (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) r c. Qed.

(* every energy condition, for every observer, at every point of the wall *)
Theorem shell_energy_conditions r c : 1 <= r <= 2 -> -1 <= c <= 1 -> energy_conditions (tE r c) (ta r c) (td r c) (tS r c) (tx r c) (tT r c) (tP r c).
Proof.
  intros Hr Hc.
  apply (certified (tE r c) (ta r c) (td r c) (tS r c) (tx r c) (tT r c) (tP r c) (mu r) (lam r)).
  - pose proof (row_n0 r c Hr Hc). lra.
  - pose proof (row_n1 r c Hr Hc). lra.
  - pose proof (row_n2 r c Hr Hc). lra.
  - pose proof (row_n3 r c Hr Hc). lra.
  - pose proof (row_m r c Hr Hc). lra.
  - pose proof (row_s r c Hr Hc). lra.
  - pose proof (row_d0 r c Hr Hc). lra.
  - pose proof (row_d1 r c Hr Hc). lra.
  - pose proof (row_d2 r c Hr Hc). lra.
  - pose proof (row_d3 r c Hr Hc). lra.
  - pose proof (row_l r c Hr Hc). lra.
Qed.

(* the radial jets are derivatives of one another *)
Lemma j_N_derive r : 1 <= r <= 2 -> is_derive j_N r (j_N_r r).
Proof.
  intros Hr. unfold j_N, j_N_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_N_r_derive r : 1 <= r <= 2 -> is_derive j_N_r r (j_N_rr r).
Proof.
  intros Hr. unfold j_N_r, j_N_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_psi_derive r : 1 <= r <= 2 -> is_derive j_psi r (j_psi_r r).
Proof.
  intros Hr. unfold j_psi, j_psi_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_psi_r_derive r : 1 <= r <= 2 -> is_derive j_psi_r r (j_psi_rr r).
Proof.
  intros Hr. unfold j_psi_r, j_psi_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_f_derive r : 1 <= r <= 2 -> is_derive j_f r (j_f_r r).
Proof.
  intros Hr. unfold j_f, j_f_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_f_r_derive r : 1 <= r <= 2 -> is_derive j_f_r r (j_f_rr r).
Proof.
  intros Hr. unfold j_f_r, j_f_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

End Drag.

(* Rocket shell on 1 <= r <= 2 with E = 1 - 1/5 (2 - r), acceleration 1/200, exhaust factor 5/4.

   G_* are the components of 8 pi T for the rocket family in terms of the jets of its profile
   functions.  Proved here: the components on this shell, the null, weak, strong and dominant energy
   conditions for every observer at every point of the wall, and that the radial jets are derivatives
   of one another.  The jets that vanish on this shell are zero in G_*. *)
Module Rocket.

Definition G_kk (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  (2 * E_r) / (E * r).

Definition G_kl (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  (((-1) * (U_r ^ 2) * (r ^ 4)) + (8 * m_r * (E ^ 2)) + ((U_r ^ 2) * (c ^ 2) * (r ^ 4)) + ((-8) * W * c * (E ^ 2)) + ((-4) * E * E_r * r) + (8 * E * E_r * m) + ((-8) * W_r * c * r * (E ^ 2)) + (4 * E * U_r * c * (r ^ 2)) + (16 * E * U * c * r) + ((-8) * E * E_r * W * c * r)) / (4 * (E ^ 2) * (r ^ 2)).

Definition G_ll (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  ((E_r * (E ^ 2) * (r ^ 2)) + ((-4) * m_u * r * (E ^ 2)) + (4 * E * (U ^ 2) * (r ^ 3)) + (4 * E_r * (E ^ 2) * (m ^ 2)) + ((-8) * E * (U ^ 2) * (c ^ 2) * (r ^ 3)) + ((-4) * E_r * m * r * (E ^ 2)) + ((-4) * U * W * (E ^ 2) * (r ^ 2)) + ((-4) * W * c * r * (E ^ 3)) + ((-2) * U_r * W * (E ^ 2) * (r ^ 3)) + (2 * U_r * c * (E ^ 2) * (r ^ 3)) + (4 * E_u * U * c * (r ^ 3)) + (4 * U * c * (E ^ 2) * (r ^ 2)) + (4 * W_u * c * (E ^ 2) * (r ^ 2)) + ((-12) * U * c * m * r * (E ^ 2)) + ((-4) * E * E_r * U * c * (r ^ 3)) + ((-4) * U * W_r * (E ^ 2) * (c ^ 2) * (r ^ 3)) + ((-4) * U_r * c * m * (E ^ 2) * (r ^ 2)) + (4 * E_r * W * c * (E ^ 2) * (r ^ 2)) + (4 * E_r * (E ^ 2) * (W ^ 2) * (c ^ 2) * (r ^ 2)) + (4 * U * c * m_r * (E ^ 2) * (r ^ 2)) + (6 * U_r * W * (E ^ 2) * (c ^ 2) * (r ^ 3)) + (12 * U * W * (E ^ 2) * (c ^ 2) * (r ^ 2)) + ((-8) * E * E_r * U * W * (c ^ 2) * (r ^ 3)) + ((-8) * E_r * W * c * m * r * (E ^ 2)) + (8 * E * E_r * U * c * m * (r ^ 2))) / (2 * (E ^ 3) * (r ^ 3)).

Definition G_k2 (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  ((4 * E * U_r * s) + (E * U_rr * r * s) + ((-1) * E_r * U_r * r * s)) / (2 * (E ^ 2)).

Definition G_l2 (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  (((-4) * W_r * s * (E ^ 3)) + (4 * U * s * (E ^ 2)) + (U_rr * s * (E ^ 2) * (r ^ 2)) + ((-8) * E_r * W * s * (E ^ 2)) + ((-8) * U_r * m * s * (E ^ 2)) + (2 * E_u * U_r * s * (r ^ 2)) + (4 * U_r * r * s * (E ^ 2)) + ((-1) * E * E_r * U_r * s * (r ^ 2)) + ((-2) * U_rr * m * r * s * (E ^ 2)) + ((-8) * E * U * U_r * c * s * (r ^ 2)) + (2 * E * E_r * U_r * m * r * s) + (2 * U_rr * W * c * s * (E ^ 2) * (r ^ 2)) + (8 * U_r * W * c * r * s * (E ^ 2)) + ((-2) * E * E_r * U_r * W * c * s * (r ^ 2))) / (4 * r * (E ^ 3)).

Definition G_22 (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  (((-4) * E * E_ru * (r ^ 2)) + ((-4) * m_rr * r * (E ^ 3)) + ((-3) * E * (U_r ^ 2) * (r ^ 4)) + (4 * E_r * E_u * (r ^ 2)) + (4 * E_r * m * (E ^ 2)) + (4 * E_r * r * (E ^ 2)) + ((-12) * E_r * m_r * r * (E ^ 2)) + ((-8) * U * c * r * (E ^ 2)) + ((-4) * U_r * c * (E ^ 2) * (r ^ 2)) + (3 * E * (U_r ^ 2) * (c ^ 2) * (r ^ 4)) + (4 * W_rr * c * (E ^ 3) * (r ^ 2)) + (8 * W_r * c * r * (E ^ 3)) + (8 * E_r * W * c * r * (E ^ 2)) + (12 * E_r * W_r * c * (E ^ 2) * (r ^ 2))) / (4 * (E ^ 3) * (r ^ 2)).

Definition G_33 (r c s E E_r E_u E_ru m m_r m_u m_rr m_ru W W_r W_u W_rr W_ru U U_r U_rr : R) : R :=
  (((-1) * E * (U_r ^ 2) * (r ^ 4)) + ((-4) * E * E_ru * (r ^ 2)) + ((-4) * m_rr * r * (E ^ 3)) + (4 * E_r * E_u * (r ^ 2)) + (4 * E_r * m * (E ^ 2)) + (4 * E_r * r * (E ^ 2)) + (E * (U_r ^ 2) * (c ^ 2) * (r ^ 4)) + ((-12) * E_r * m_r * r * (E ^ 2)) + ((-8) * U * c * r * (E ^ 2)) + ((-4) * U_r * c * (E ^ 2) * (r ^ 2)) + (4 * W_rr * c * (E ^ 3) * (r ^ 2)) + (8 * W_r * c * r * (E ^ 3)) + (8 * E_r * W * c * r * (E ^ 2)) + (12 * E_r * W_r * c * (E ^ 2) * (r ^ 2))) / (4 * (E ^ 3) * (r ^ 2)).

Definition j_E (r : R) : R := (4 + (r - 1)) / 5.
Definition j_E_r (r : R) : R := 1 / 5.
Definition j_E_u (r : R) : R := (123 + ((-123) * (r - 1))) / 29600.
Definition j_E_rr (r : R) : R := 0.
Definition j_E_ru (r : R) : R := (-123) / 29600.
Definition j_e (r : R) : R := 0.
Definition j_e_r (r : R) : R := 0.
Definition j_e_u (r : R) : R := 0.
Definition j_e_rr (r : R) : R := 0.
Definition j_e_ru (r : R) : R := 0.
Definition j_m (r : R) : R := ((r - 1) * (24 + ((r - 1) * (15 + (2 * (r - 1)))))) / (6 * ((4 + (r - 1)) ^ 2)).
Definition j_m_r (r : R) : R := (48 + ((r - 1) * (48 + ((r - 1) * (12 + (r - 1)))))) / (3 * ((4 + (r - 1)) ^ 3)).
Definition j_m_u (r : R) : R := ((r - 1) * ((-984) + ((r - 1) * ((-492) + ((-41) * (r - 1)))))) / (2368 * ((4 + (r - 1)) ^ 3)).
Definition j_m_rr (r : R) : R := 16 / ((4 + (r - 1)) ^ 4).
Definition j_m_ru (r : R) : R := ((-246) + ((-123) * (r - 1))) / (148 * ((4 + (r - 1)) ^ 4)).
Definition j_W (r : R) : R := (((r - 1) ^ 2) * (5 + ((-3) * (r - 1)))) / (8 * ((4 + (r - 1)) ^ 2)).
Definition j_W_r (r : R) : R := ((r - 1) * (40 + ((r - 1) * ((-36) + ((-3) * (r - 1)))))) / (8 * ((4 + (r - 1)) ^ 3)).
Definition j_W_u (r : R) : R := (((r - 1) ^ 2) * ((-615) + ((r - 1) * (984 + ((-369) * (r - 1)))))) / (23680 * ((4 + (r - 1)) ^ 3)).
Definition j_W_rr (r : R) : R := (20 + ((-46) * (r - 1))) / ((4 + (r - 1)) ^ 4).
Definition j_W_ru (r : R) : R := ((r - 1) * ((-4920) + ((r - 1) * (12423 + ((r - 1) * ((-5904) + ((-369) * (r - 1)))))))) / (23680 * ((4 + (r - 1)) ^ 4)).
Definition j_U (r : R) : R := (((r - 1) ^ 2) * (3 + ((-2) * (r - 1)))) / 200.
Definition j_U_r (r : R) : R := ((r - 1) * (3 + ((-3) * (r - 1)))) / 100.
Definition j_U_u (r : R) : R := 0.
Definition j_U_rr (r : R) : R := (3 + ((-6) * (r - 1))) / 100.
Definition j_U_ru (r : R) : R := 0.

Definition x_kk (r c s : R) : R := (2 / (r * (4 + (r - 1)))).

Definition x_kl (r c s : R) : R := ((76800 + ((r - 1) * (76800 + ((-36480) * c) + ((r - 1) * (19092 + (c * (5520 + (108 * c))) + ((r - 1) * (1357 + (c * (66960 + (243 * c))) + ((r - 1) * (54 + (c * ((-15840) + ((-54) * c))) + ((r - 1) * (459 + (c * ((-13680) + ((-459) * c))) + ((r - 1) * (216 + (c * ((-1680) + ((-216) * c))) + ((r - 1) * ((-189) + (189 * (c ^ 2)) + ((r - 1) * ((-162) + (162 * (c ^ 2)) + ((r - 1) * ((-27) + (27 * (c ^ 2))))))))))))))))))))) / (4800 * (r ^ 2) * ((4 + (r - 1)) ^ 3))).

Definition x_ll (r c s : R) : R := ((136396800 + ((r - 1) * (290505600 + (40919040 * c) + ((r - 1) * (235591200 + (78439320 * c) + ((r - 1) * (88519200 + (c * (50063850 + (1598400 * c))) + ((r - 1) * (13476372 + (c * ((-36861615) + (7023636 * c))) + ((r - 1) * (2187384 + (c * ((-87428580) + (1799532 * c))) + ((r - 1) * (5877376 + (c * ((-48340680) + ((-10097892) * c))) + ((r - 1) * (1630368 + (c * ((-11459700) + ((-4126536) * c))) + ((r - 1) * ((-2091240) + (c * ((-1184625) + (3643020 * c))) + ((r - 1) * ((-677988) + (c * ((-42210) + (1122876 * c))) + ((r - 1) * (135864 + ((-291708) * (c ^ 2)) + ((r - 1) * (63936 + ((-127872) * (c ^ 2)) + ((r - 1) * (5328 + ((-10656) * (c ^ 2))))))))))))))))))))))))))) / (1065600 * (r ^ 3) * ((4 + (r - 1)) ^ 5))).

Definition x_k2 (r c s : R) : R := s * ((12 + ((r - 1) * (36 + ((r - 1) * ((-63) + ((-15) * (r - 1))))))) / (40 * ((4 + (r - 1)) ^ 2))).

Definition x_l2 (r c s : R) : R := s * ((568320 + ((r - 1) * ((-2455340) + ((r - 1) * (1896245 + (44400 * c) + ((r - 1) * ((-1340840) + (65712 * c) + ((r - 1) * ((-1513290) + ((-277500) * c) + ((r - 1) * ((-356700) + ((-106116) * c) + ((r - 1) * ((-24795) + (215340 * c) + ((r - 1) * (((-444) * c) + ((r - 1) * (((-26640) * c) + ((-3552) * c * (r - 1))))))))))))))))))) / (236800 * ((4 + (r - 1)) ^ 4) * r)).

Definition x_22 (r c s : R) : R := ((92250 + (888000 * c) + ((r - 1) * (894900 + (1527360 * c) + ((r - 1) * (524262 + (c * ((-3174600) + (11988 * c))) + ((r - 1) * (32227 + (c * ((-3809520) + (26973 * c))) + ((r - 1) * (5994 + (c * (333000 + ((-5994) * c))) + ((r - 1) * (50949 + (c * (372960 + ((-50949) * c))) + ((r - 1) * (23976 + (c * (44400 + ((-23976) * c))) + ((r - 1) * ((-20979) + (20979 * (c ^ 2)) + ((r - 1) * ((-17982) + (17982 * (c ^ 2)) + ((r - 1) * ((-2997) + (2997 * (c ^ 2))))))))))))))))))))) / (177600 * (r ^ 2) * ((4 + (r - 1)) ^ 3))).

Definition x_33 (r c s : R) : R := ((92250 + (888000 * c) + ((r - 1) * (894900 + (1527360 * c) + ((r - 1) * (532254 + (c * ((-3174600) + (3996 * c))) + ((r - 1) * (50209 + (c * ((-3809520) + (8991 * c))) + ((r - 1) * (1998 + (c * (333000 + ((-1998) * c))) + ((r - 1) * (16983 + (c * (372960 + ((-16983) * c))) + ((r - 1) * (7992 + (c * (44400 + ((-7992) * c))) + ((r - 1) * ((-6993) + (6993 * (c ^ 2)) + ((r - 1) * ((-5994) + (5994 * (c ^ 2)) + ((r - 1) * ((-999) + (999 * (c ^ 2))))))))))))))))))))) / (177600 * (r ^ 2) * ((4 + (r - 1)) ^ 3))).

(* field's expressions of the jets, of which those of the identities below are built *)
Definition fe_E : FExpr BinNums.Z := (FEdiv (FEadd (FEc 4%Z) (FEX _ 1)) (FEc 5%Z)).
Definition fe_E_r : FExpr BinNums.Z := (FEdiv (FEc 1%Z) (FEc 5%Z)).
Definition fe_E_u : FExpr BinNums.Z := (FEdiv (FEadd (FEc 123%Z) (FEmul (FEc (-123)%Z) (FEX _ 1))) (FEc 29600%Z)).
Definition fe_E_ru : FExpr BinNums.Z := (FEdiv (FEc (-123)%Z) (FEc 29600%Z)).
Definition fe_m : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc 24%Z) (FEmul (FEX _ 1) (FEadd (FEc 15%Z) (FEmul (FEc 2%Z) (FEX _ 1)))))) (FEmul (FEc 6%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 2)))).
Definition fe_m_r : FExpr BinNums.Z := (FEdiv (FEadd (FEc 48%Z) (FEmul (FEX _ 1) (FEadd (FEc 48%Z) (FEmul (FEX _ 1) (FEadd (FEc 12%Z) (FEX _ 1)))))) (FEmul (FEc 3%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Definition fe_m_u : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc (-984)%Z) (FEmul (FEX _ 1) (FEadd (FEc (-492)%Z) (FEmul (FEc (-41)%Z) (FEX _ 1)))))) (FEmul (FEc 2368%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Definition fe_m_rr : FExpr BinNums.Z := (FEdiv (FEc 16%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 4))).
Definition fe_m_ru : FExpr BinNums.Z := (FEdiv (FEadd (FEc (-246)%Z) (FEmul (FEc (-123)%Z) (FEX _ 1))) (FEmul (FEc 148%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 4)))).
Definition fe_W : FExpr BinNums.Z := (FEdiv (FEmul (FEpow (FEX _ 1) (Npos 2)) (FEadd (FEc 5%Z) (FEmul (FEc (-3)%Z) (FEX _ 1)))) (FEmul (FEc 8%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 2)))).
Definition fe_W_r : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc 40%Z) (FEmul (FEX _ 1) (FEadd (FEc (-36)%Z) (FEmul (FEc (-3)%Z) (FEX _ 1)))))) (FEmul (FEc 8%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Definition fe_W_u : FExpr BinNums.Z := (FEdiv (FEmul (FEpow (FEX _ 1) (Npos 2)) (FEadd (FEc (-615)%Z) (FEmul (FEX _ 1) (FEadd (FEc 984%Z) (FEmul (FEc (-369)%Z) (FEX _ 1)))))) (FEmul (FEc 23680%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Definition fe_W_rr : FExpr BinNums.Z := (FEdiv (FEadd (FEc 20%Z) (FEmul (FEc (-46)%Z) (FEX _ 1))) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 4))).
Definition fe_W_ru : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc (-4920)%Z) (FEmul (FEX _ 1) (FEadd (FEc 12423%Z) (FEmul (FEX _ 1) (FEadd (FEc (-5904)%Z) (FEmul (FEc (-369)%Z) (FEX _ 1)))))))) (FEmul (FEc 23680%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 4)))).
Definition fe_U : FExpr BinNums.Z := (FEdiv (FEmul (FEpow (FEX _ 1) (Npos 2)) (FEadd (FEc 3%Z) (FEmul (FEc (-2)%Z) (FEX _ 1)))) (FEc 200%Z)).
Definition fe_U_r : FExpr BinNums.Z := (FEdiv (FEmul (FEX _ 1) (FEadd (FEc 3%Z) (FEmul (FEc (-3)%Z) (FEX _ 1)))) (FEc 100%Z)).
Definition fe_U_rr : FExpr BinNums.Z := (FEdiv (FEadd (FEc 3%Z) (FEmul (FEc (-6)%Z) (FEX _ 1))) (FEc 100%Z)).

Definition fe_G_kk : FExpr BinNums.Z := (FEdiv (FEmul (FEc 2%Z) fe_E_r) (FEmul fe_E (FEX _ 2))).
Definition fe_x_kk : FExpr BinNums.Z := (FEdiv (FEc 2%Z) (FEmul (FEX _ 2) (FEadd (FEc 4%Z) (FEX _ 1)))).
Lemma x_kk_closed r c s : 1 <= r <= 2 -> G_kk r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_kk r c s.
Proof. intros Hr. identity r c s fe_G_kk fe_x_kk. Qed.

Definition fe_G_kl : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEc (-1)%Z) (FEpow fe_U_r (Npos 2))) (FEpow (FEX _ 2) (Npos 4))) (FEmul (FEmul (FEc 8%Z) fe_m_r) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEpow fe_U_r (Npos 2)) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 4)))) (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_W) (FEX _ 3)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_E) fe_E_r) (FEX _ 2))) (FEmul (FEmul (FEmul (FEc 8%Z) fe_E) fe_E_r) fe_m)) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_W_r) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_E) fe_U_r) (FEX _ 3)) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 16%Z) fe_E) fe_U) (FEX _ 3)) (FEX _ 2))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E) fe_E_r) fe_W) (FEX _ 3)) (FEX _ 2))) (FEmul (FEmul (FEc 4%Z) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))).
Definition fe_x_kl : FExpr BinNums.Z := (FEdiv (FEadd (FEc 76800%Z) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 76800%Z) (FEmul (FEc (-36480)%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 19092%Z) (FEmul (FEX _ 3) (FEadd (FEc 5520%Z) (FEmul (FEc 108%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 1357%Z) (FEmul (FEX _ 3) (FEadd (FEc 66960%Z) (FEmul (FEc 243%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 54%Z) (FEmul (FEX _ 3) (FEadd (FEc (-15840)%Z) (FEmul (FEc (-54)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 459%Z) (FEmul (FEX _ 3) (FEadd (FEc (-13680)%Z) (FEmul (FEc (-459)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 216%Z) (FEmul (FEX _ 3) (FEadd (FEc (-1680)%Z) (FEmul (FEc (-216)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-189)%Z) (FEmul (FEc 189%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-162)%Z) (FEmul (FEc 162%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-27)%Z) (FEmul (FEc 27%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))) (FEmul (FEmul (FEc 4800%Z) (FEpow (FEX _ 2) (Npos 2))) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Lemma x_kl_closed r c s : 1 <= r <= 2 -> G_kl r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_kl r c s.
Proof. intros Hr. identity r c s fe_G_kl fe_x_kl. Qed.

Definition fe_G_ll : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul fe_E_r (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2))) (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_m_u) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E) (FEpow fe_U (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) (FEpow fe_E (Npos 2))) (FEpow fe_m (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E) (FEpow fe_U (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_E_r) fe_m) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_U) fe_W) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_W) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc (-2)%Z) fe_U_r) fe_W) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc 2%Z) fe_U_r) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_u) fe_U) (FEX _ 3)) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_U) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_W_u) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-12)%Z) fe_U) (FEX _ 3)) fe_m) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_E) fe_E_r) fe_U) (FEX _ 3)) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_U) fe_W_r) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_U_r) (FEX _ 3)) fe_m) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) fe_W) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) (FEpow fe_E (Npos 2))) (FEpow fe_W (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_U) (FEX _ 3)) fe_m_r) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 6%Z) fe_U_r) fe_W) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 12%Z) fe_U) fe_W) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E) fe_E_r) fe_U) fe_W) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E_r) fe_W) (FEX _ 3)) fe_m) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_E) fe_E_r) fe_U) (FEX _ 3)) fe_m) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEc 2%Z) (FEpow fe_E (Npos 3))) (FEpow (FEX _ 2) (Npos 3)))).
Definition fe_x_ll : FExpr BinNums.Z := (FEdiv (FEadd (FEc 136396800%Z) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 290505600%Z) (FEmul (FEc 40919040%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 235591200%Z) (FEmul (FEc 78439320%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 88519200%Z) (FEmul (FEX _ 3) (FEadd (FEc 50063850%Z) (FEmul (FEc 1598400%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 13476372%Z) (FEmul (FEX _ 3) (FEadd (FEc (-36861615)%Z) (FEmul (FEc 7023636%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 2187384%Z) (FEmul (FEX _ 3) (FEadd (FEc (-87428580)%Z) (FEmul (FEc 1799532%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 5877376%Z) (FEmul (FEX _ 3) (FEadd (FEc (-48340680)%Z) (FEmul (FEc (-10097892)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 1630368%Z) (FEmul (FEX _ 3) (FEadd (FEc (-11459700)%Z) (FEmul (FEc (-4126536)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-2091240)%Z) (FEmul (FEX _ 3) (FEadd (FEc (-1184625)%Z) (FEmul (FEc 3643020%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-677988)%Z) (FEmul (FEX _ 3) (FEadd (FEc (-42210)%Z) (FEmul (FEc 1122876%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 135864%Z) (FEmul (FEc (-291708)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 63936%Z) (FEmul (FEc (-127872)%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc 5328%Z) (FEmul (FEc (-10656)%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))))))))) (FEmul (FEmul (FEc 1065600%Z) (FEpow (FEX _ 2) (Npos 3))) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 5)))).
Lemma x_ll_closed r c s : 1 <= r <= 2 -> G_ll r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_ll r c s.
Proof. intros Hr. identity r c s fe_G_ll fe_x_ll. Qed.

Definition fe_G_k2 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEmul (FEmul (FEmul (FEc 4%Z) fe_E) fe_U_r) (FEX _ 4)) (FEmul (FEmul (FEmul fe_E fe_U_rr) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_E_r) fe_U_r) (FEX _ 2)) (FEX _ 4))) (FEmul (FEc 2%Z) (FEpow fe_E (Npos 2)))).
Definition fe_x_k2 : FExpr BinNums.Z := (FEmul (FEX _ 4) (FEdiv (FEadd (FEc 12%Z) (FEmul (FEX _ 1) (FEadd (FEc 36%Z) (FEmul (FEX _ 1) (FEadd (FEc (-63)%Z) (FEmul (FEc (-15)%Z) (FEX _ 1))))))) (FEmul (FEc 40%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 2))))).
Lemma x_k2_closed r c s : 1 <= r <= 2 -> G_k2 r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_k2 r c s.
Proof. intros Hr. identity r c s fe_G_k2 fe_x_k2. Qed.

Definition fe_G_l2 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_W_r) (FEX _ 4)) (FEpow fe_E (Npos 3))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_U) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul fe_U_rr (FEX _ 4)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E_r) fe_W) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_U_r) fe_m) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 2%Z) fe_E_u) fe_U_r) (FEX _ 4)) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_U_r) (FEX _ 2)) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_E) fe_E_r) fe_U_r) (FEX _ 4)) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-2)%Z) fe_U_rr) fe_m) (FEX _ 2)) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_E) fe_U) fe_U_r) (FEX _ 3)) (FEX _ 4)) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 2%Z) fe_E) fe_E_r) fe_U_r) fe_m) (FEX _ 2)) (FEX _ 4))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 2%Z) fe_U_rr) fe_W) (FEX _ 3)) (FEX _ 4)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_U_r) fe_W) (FEX _ 3)) (FEX _ 2)) (FEX _ 4)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEmul (FEc (-2)%Z) fe_E) fe_E_r) fe_U_r) fe_W) (FEX _ 3)) (FEX _ 4)) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEc 4%Z) (FEX _ 2)) (FEpow fe_E (Npos 3)))).
Definition fe_x_l2 : FExpr BinNums.Z := (FEmul (FEX _ 4) (FEdiv (FEadd (FEc 568320%Z) (FEmul (FEX _ 1) (FEadd (FEc (-2455340)%Z) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 1896245%Z) (FEmul (FEc 44400%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1340840)%Z) (FEmul (FEc 65712%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-1513290)%Z) (FEmul (FEc (-277500)%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-356700)%Z) (FEmul (FEc (-106116)%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-24795)%Z) (FEmul (FEc 215340%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-444)%Z) (FEX _ 3)) (FEmul (FEX _ 1) (FEadd (FEmul (FEc (-26640)%Z) (FEX _ 3)) (FEmul (FEmul (FEc (-3552)%Z) (FEX _ 3)) (FEX _ 1))))))))))))))))))) (FEmul (FEmul (FEc 236800%Z) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 4))) (FEX _ 2)))).
Lemma x_l2_closed r c s : 1 <= r <= 2 -> G_l2 r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_l2 r c s.
Proof. intros Hr. identity r c s fe_G_l2 fe_x_l2. Qed.

Definition fe_G_22 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_E) fe_E_ru) (FEpow (FEX _ 2) (Npos 2))) (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_m_rr) (FEX _ 2)) (FEpow fe_E (Npos 3)))) (FEmul (FEmul (FEmul (FEc (-3)%Z) fe_E) (FEpow fe_U_r (Npos 2))) (FEpow (FEX _ 2) (Npos 4)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) fe_E_u) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) fe_m) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-12)%Z) fe_E_r) fe_m_r) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_U) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_U_r) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 3%Z) fe_E) (FEpow fe_U_r (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 4)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_W_rr) (FEX _ 3)) (FEpow fe_E (Npos 3))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_W_r) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_E_r) fe_W) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 12%Z) fe_E_r) fe_W_r) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEc 4%Z) (FEpow fe_E (Npos 3))) (FEpow (FEX _ 2) (Npos 2)))).
Definition fe_x_22 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEc 92250%Z) (FEmul (FEc 888000%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 894900%Z) (FEmul (FEc 1527360%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 524262%Z) (FEmul (FEX _ 3) (FEadd (FEc (-3174600)%Z) (FEmul (FEc 11988%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 32227%Z) (FEmul (FEX _ 3) (FEadd (FEc (-3809520)%Z) (FEmul (FEc 26973%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 5994%Z) (FEmul (FEX _ 3) (FEadd (FEc 333000%Z) (FEmul (FEc (-5994)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 50949%Z) (FEmul (FEX _ 3) (FEadd (FEc 372960%Z) (FEmul (FEc (-50949)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 23976%Z) (FEmul (FEX _ 3) (FEadd (FEc 44400%Z) (FEmul (FEc (-23976)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-20979)%Z) (FEmul (FEc 20979%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-17982)%Z) (FEmul (FEc 17982%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-2997)%Z) (FEmul (FEc 2997%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))) (FEmul (FEmul (FEc 177600%Z) (FEpow (FEX _ 2) (Npos 2))) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Lemma x_22_closed r c s : 1 <= r <= 2 -> G_22 r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_22 r c s.
Proof. intros Hr. identity r c s fe_G_22 fe_x_22. Qed.

Definition fe_G_33 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEadd (FEmul (FEmul (FEmul (FEc (-1)%Z) fe_E) (FEpow fe_U_r (Npos 2))) (FEpow (FEX _ 2) (Npos 4))) (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_E) fe_E_ru) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_m_rr) (FEX _ 2)) (FEpow fe_E (Npos 3)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) fe_E_u) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) fe_m) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEc 4%Z) fe_E_r) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul fe_E (FEpow fe_U_r (Npos 2))) (FEpow (FEX _ 3) (Npos 2))) (FEpow (FEX _ 2) (Npos 4)))) (FEmul (FEmul (FEmul (FEmul (FEc (-12)%Z) fe_E_r) fe_m_r) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-8)%Z) fe_U) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc (-4)%Z) fe_U_r) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 4%Z) fe_W_rr) (FEX _ 3)) (FEpow fe_E (Npos 3))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_W_r) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 3)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 8%Z) fe_E_r) fe_W) (FEX _ 3)) (FEX _ 2)) (FEpow fe_E (Npos 2)))) (FEmul (FEmul (FEmul (FEmul (FEmul (FEc 12%Z) fe_E_r) fe_W_r) (FEX _ 3)) (FEpow fe_E (Npos 2))) (FEpow (FEX _ 2) (Npos 2)))) (FEmul (FEmul (FEc 4%Z) (FEpow fe_E (Npos 3))) (FEpow (FEX _ 2) (Npos 2)))).
Definition fe_x_33 : FExpr BinNums.Z := (FEdiv (FEadd (FEadd (FEc 92250%Z) (FEmul (FEc 888000%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 894900%Z) (FEmul (FEc 1527360%Z) (FEX _ 3))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 532254%Z) (FEmul (FEX _ 3) (FEadd (FEc (-3174600)%Z) (FEmul (FEc 3996%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 50209%Z) (FEmul (FEX _ 3) (FEadd (FEc (-3809520)%Z) (FEmul (FEc 8991%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 1998%Z) (FEmul (FEX _ 3) (FEadd (FEc 333000%Z) (FEmul (FEc (-1998)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 16983%Z) (FEmul (FEX _ 3) (FEadd (FEc 372960%Z) (FEmul (FEc (-16983)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc 7992%Z) (FEmul (FEX _ 3) (FEadd (FEc 44400%Z) (FEmul (FEc (-7992)%Z) (FEX _ 3))))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-6993)%Z) (FEmul (FEc 6993%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEadd (FEc (-5994)%Z) (FEmul (FEc 5994%Z) (FEpow (FEX _ 3) (Npos 2)))) (FEmul (FEX _ 1) (FEadd (FEc (-999)%Z) (FEmul (FEc 999%Z) (FEpow (FEX _ 3) (Npos 2)))))))))))))))))))))) (FEmul (FEmul (FEc 177600%Z) (FEpow (FEX _ 2) (Npos 2))) (FEpow (FEadd (FEc 4%Z) (FEX _ 1)) (Npos 3)))).
Lemma x_33_closed r c s : 1 <= r <= 2 -> G_33 r c s (j_E r) (j_E_r r) (j_E_u r) (j_E_ru r) (j_m r) (j_m_r r) (j_m_u r) (j_m_rr r) (j_m_ru r) (j_W r) (j_W_r r) (j_W_u r) (j_W_rr r) (j_W_ru r) (j_U r) (j_U_r r) (j_U_rr r) = x_33 r c s.
Proof. intros Hr. identity r c s fe_G_33 fe_x_33. Qed.

(* the frame is (q k + l/q)/sqrt2, (q k - l/q)/sqrt2, e_th, e_ph with q^2 = H0/2 *)
Definition H0 (r : R) : R := (48 + ((r - 1) * (48 + ((r - 1) * (12 + (r - 1)))))) / (3 * ((4 + (r - 1)) ^ 2) * r).

Lemma H0_positive r : 1 <= r <= 2 -> 0 < H0 r.
Proof. intros Hr. unfold H0. interval. Qed.

(* the components at radius r and c = cos(theta), with sin(theta) = sqrt (1 - c^2) *)
Definition tE (r c : R) : R := (H0 r * (x_kk r c (sqrt (1 - c * c))) / 2 + 2 * (x_kl r c (sqrt (1 - c * c))) + 2 * (x_ll r c (sqrt (1 - c * c))) / H0 r) / 2.
Definition ta (r c : R) : R := (H0 r * (x_kk r c (sqrt (1 - c * c))) / 2 - 2 * (x_ll r c (sqrt (1 - c * c))) / H0 r) / 2.
Definition td (r c : R) : R := (H0 r * (x_k2 r c (sqrt (1 - c * c))) / 2 + x_l2 r c (sqrt (1 - c * c))) / sqrt (H0 r).
Definition tS (r c : R) : R := (H0 r * (x_kk r c (sqrt (1 - c * c))) / 2 - 2 * (x_kl r c (sqrt (1 - c * c))) + 2 * (x_ll r c (sqrt (1 - c * c))) / H0 r) / 2.
Definition tx (r c : R) : R := (H0 r * (x_k2 r c (sqrt (1 - c * c))) / 2 - (x_l2 r c (sqrt (1 - c * c)))) / sqrt (H0 r).
Definition tT (r c : R) : R := x_22 r c (sqrt (1 - c * c)).
Definition tP (r c : R) : R := x_33 r c (sqrt (1 - c * c)).

Definition rho (r : R) : R := (96 + ((r - 1) * (96 + ((r - 1) * (24 + (2 * (r - 1))))))) / (3 * (r ^ 2) * ((4 + (r - 1)) ^ 3)).
Definition mu (r : R) : R := (3 / 4) * rho r.
Definition lam (r : R) : R := (1 / 2) * (rho r * rho r).

(* the trees of rho and of the components, of which the rows' trees are built *)
Definition tree_rho : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 96)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 96)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 24)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))).
Definition tree_H0 : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 48)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 48)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 12)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))) (Tree.Evar 0))).
Definition tree_E : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul tree_H0 (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Mul (Tree.Evar 0) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))) (Tree.Econst (Tree.Int 2))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 76800)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 76800)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-36480))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 19092)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5520)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 108)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1357)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 66960)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 243)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 54)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-15840))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-54))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 459)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-13680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-459))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 216)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-216))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-189))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 189)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-162))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 162)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-27))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 27)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 4800)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 136396800)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 290505600)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 40919040)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 235591200)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 78439320)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 88519200)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 50063850)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1598400)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 13476372)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-36861615))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 7023636)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2187384)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-87428580))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1799532)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5877376)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-48340680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10097892))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1630368)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-11459700))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-4126536))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2091240))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1184625))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3643020)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-677988))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-42210))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1122876)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 135864)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-291708))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 63936)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-127872))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5328)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10656))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1065600)) (Tree.Eunary (Tree.PowerInt 3) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) tree_H0)) (Tree.Econst (Tree.Int 2))).
Definition tree_a : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul tree_H0 (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Mul (Tree.Evar 0) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))) (Tree.Econst (Tree.Int 2))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 136396800)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 290505600)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 40919040)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 235591200)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 78439320)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 88519200)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 50063850)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1598400)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 13476372)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-36861615))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 7023636)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2187384)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-87428580))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1799532)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5877376)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-48340680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10097892))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1630368)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-11459700))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-4126536))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2091240))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1184625))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3643020)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-677988))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-42210))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1122876)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 135864)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-291708))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 63936)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-127872))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5328)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10656))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1065600)) (Tree.Eunary (Tree.PowerInt 3) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) tree_H0)) (Tree.Econst (Tree.Int 2))).
Definition tree_d : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul tree_H0 (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 12)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 36)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-63))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-15))) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 40)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Econst (Tree.Int 2))) (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 568320)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2455340))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1896245)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 44400)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1340840))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 65712)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1513290))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-277500))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-356700))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-106116))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-24795))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 215340)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-444))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-26640))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-3552))) (Tree.Evar 1)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 236800)) (Tree.Eunary (Tree.PowerInt 4) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))) (Tree.Evar 0))))) (Tree.Eunary Tree.Sqrt tree_H0)).
Definition tree_S : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul tree_H0 (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Mul (Tree.Evar 0) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))) (Tree.Econst (Tree.Int 2))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 76800)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 76800)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-36480))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 19092)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5520)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 108)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1357)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 66960)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 243)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 54)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-15840))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-54))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 459)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-13680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-459))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 216)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-216))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-189))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 189)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-162))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 162)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-27))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 27)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 4800)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2)) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 136396800)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 290505600)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 40919040)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 235591200)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 78439320)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 88519200)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 50063850)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1598400)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 13476372)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-36861615))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 7023636)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 2187384)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-87428580))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1799532)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5877376)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-48340680))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10097892))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1630368)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-11459700))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-4126536))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2091240))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1184625))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3643020)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-677988))) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-42210))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1122876)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 135864)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-291708))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 63936)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-127872))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5328)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-10656))) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1065600)) (Tree.Eunary (Tree.PowerInt 3) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 5) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))))) tree_H0)) (Tree.Econst (Tree.Int 2))).
Definition tree_x : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Mul tree_H0 (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 12)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 36)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-63))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-15))) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 40)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))) (Tree.Econst (Tree.Int 2))) (Tree.Ebinary Tree.Mul (Tree.Eunary Tree.Sqrt (Tree.Ebinary Tree.Sub (Tree.Econst (Tree.Int 1)) (Tree.Eunary Tree.Sqr (Tree.Evar 1)))) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 568320)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2455340))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1896245)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 44400)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1340840))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 65712)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-1513290))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-277500))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-356700))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-106116))) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-24795))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 215340)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-444))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-26640))) (Tree.Evar 1)) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-3552))) (Tree.Evar 1)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 236800)) (Tree.Eunary (Tree.PowerInt 4) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1)))))) (Tree.Evar 0))))) (Tree.Eunary Tree.Sqrt tree_H0)).
Definition tree_T : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 92250)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 888000)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 894900)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1527360)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 524262)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-3174600))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 11988)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 32227)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-3809520))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 26973)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 5994)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 333000)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-5994))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 50949)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 372960)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-50949))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 23976)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 44400)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-23976))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-20979))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 20979)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-17982))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 17982)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-2997))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 2997)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 177600)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))).
Definition tree_P : Tree.expr := (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 92250)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 888000)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 894900)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 1527360)) (Tree.Evar 1))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 532254)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-3174600))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 3996)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 50209)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-3809520))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 8991)) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 1998)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 333000)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-1998))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 16983)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 372960)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-16983))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 7992)) (Tree.Ebinary Tree.Mul (Tree.Evar 1) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 44400)) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int (-7992))) (Tree.Evar 1))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-6993))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 6993)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-5994))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 5994)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int (-999))) (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 999)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 1)))))))))))))))))))))) (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Mul (Tree.Econst (Tree.Int 177600)) (Tree.Eunary (Tree.PowerInt 2) (Tree.Evar 0))) (Tree.Eunary (Tree.PowerInt 3) (Tree.Ebinary Tree.Add (Tree.Econst (Tree.Int 4)) (Tree.Ebinary Tree.Sub (Tree.Evar 0) (Tree.Econst (Tree.Int 1))))))).

Lemma row_n0 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tE r c - mu r - Rabs (ta r c) - Rabs (td r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub tree_E (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho)) (Tree.Eunary Tree.Abs tree_a)) (Tree.Eunary Tree.Abs tree_d)) r c. Qed.

Lemma row_n1 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tS r c + mu r - Rabs (ta r c) - Rabs (tx r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add tree_S (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho)) (Tree.Eunary Tree.Abs tree_a)) (Tree.Eunary Tree.Abs tree_x)) r c. Qed.

Lemma row_n2 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tT r c + mu r - Rabs (td r c) - Rabs (tx r c).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add tree_T (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho)) (Tree.Eunary Tree.Abs tree_d)) (Tree.Eunary Tree.Abs tree_x)) r c. Qed.

Lemma row_n3 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= tP r c + mu r.
Proof. wall (Tree.Ebinary Tree.Add tree_P (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho)) r c. Qed.

Lemma row_m r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= mu r.
Proof. wall (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho) r c. Qed.

Lemma row_s r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= mu r + trace (tE r c) (tS r c) (tT r c) (tP r c) / 2.
Proof. wall (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 3)) (Tree.Econst (Tree.Int 4))) tree_rho) (Tree.Ebinary Tree.Div (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Add (Tree.Eunary Tree.Neg tree_E) tree_S) tree_T) tree_P) (Tree.Econst (Tree.Int 2)))) r c. Qed.

Lemma row_d0 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D00 (tE r c) (ta r c) (td r c) (lam r) - Rabs (D01 (tE r c) (ta r c) (td r c) (tS r c) (tx r c)) - Rabs (D02 (tE r c) (ta r c) (td r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Eunary Tree.Neg (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho))) (Tree.Eunary Tree.Sqr tree_E)) (Tree.Eunary Tree.Sqr tree_a)) (Tree.Eunary Tree.Sqr tree_d)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_a) (Tree.Ebinary Tree.Mul tree_a tree_S)) (Tree.Ebinary Tree.Mul tree_d tree_x)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_d) (Tree.Ebinary Tree.Mul tree_a tree_x)) (Tree.Ebinary Tree.Mul tree_d tree_T)))) r c. Qed.

Lemma row_d1 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D11 (ta r c) (tS r c) (tx r c) (lam r) - Rabs (D01 (tE r c) (ta r c) (td r c) (tS r c) (tx r c)) - Rabs (D12 (ta r c) (td r c) (tS r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_a)) (Tree.Eunary Tree.Sqr tree_S)) (Tree.Eunary Tree.Sqr tree_x)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_a) (Tree.Ebinary Tree.Mul tree_a tree_S)) (Tree.Ebinary Tree.Mul tree_d tree_x)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_a tree_d) (Tree.Ebinary Tree.Mul tree_x (Tree.Ebinary Tree.Add tree_S tree_T))))) r c. Qed.

Lemma row_d2 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D22 (td r c) (tx r c) (tT r c) (lam r) - Rabs (D02 (tE r c) (ta r c) (td r c) (tx r c) (tT r c)) - Rabs (D12 (ta r c) (td r c) (tS r c) (tx r c) (tT r c)).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Add (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_d)) (Tree.Eunary Tree.Sqr tree_x)) (Tree.Eunary Tree.Sqr tree_T)) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_E tree_d) (Tree.Ebinary Tree.Mul tree_a tree_x)) (Tree.Ebinary Tree.Mul tree_d tree_T)))) (Tree.Eunary Tree.Abs (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul tree_a tree_d) (Tree.Ebinary Tree.Mul tree_x (Tree.Ebinary Tree.Add tree_S tree_T))))) r c. Qed.

Lemma row_d3 r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= D33 (tP r c) (lam r).
Proof. wall (Tree.Ebinary Tree.Sub (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) (Tree.Eunary Tree.Sqr tree_P)) r c. Qed.

Lemma row_l r c : 1 <= r <= 2 -> -1 <= c <= 1 -> 0 <= lam r.
Proof. wall (Tree.Ebinary Tree.Mul (Tree.Ebinary Tree.Div (Tree.Econst (Tree.Int 1)) (Tree.Econst (Tree.Int 2))) (Tree.Eunary Tree.Sqr tree_rho)) r c. Qed.

(* every energy condition, for every observer, at every point of the wall *)
Theorem shell_energy_conditions r c : 1 <= r <= 2 -> -1 <= c <= 1 -> energy_conditions (tE r c) (ta r c) (td r c) (tS r c) (tx r c) (tT r c) (tP r c).
Proof.
  intros Hr Hc.
  apply (certified (tE r c) (ta r c) (td r c) (tS r c) (tx r c) (tT r c) (tP r c) (mu r) (lam r)).
  - pose proof (row_n0 r c Hr Hc). lra.
  - pose proof (row_n1 r c Hr Hc). lra.
  - pose proof (row_n2 r c Hr Hc). lra.
  - pose proof (row_n3 r c Hr Hc). lra.
  - pose proof (row_m r c Hr Hc). lra.
  - pose proof (row_s r c Hr Hc). lra.
  - pose proof (row_d0 r c Hr Hc). lra.
  - pose proof (row_d1 r c Hr Hc). lra.
  - pose proof (row_d2 r c Hr Hc). lra.
  - pose proof (row_d3 r c Hr Hc). lra.
  - pose proof (row_l r c Hr Hc). lra.
Qed.

(* outside the wall the solution is Kinnersley's photon rocket: null dust whose density is a
   positive multiple of -(m_u + 3 m a c), with m_u = -3 kappa a m on this shell *)
Lemma exhaust_nonnegative m a kappa c :
  0 <= m -> 0 <= a -> 1 <= kappa -> -1 <= c <= 1 -> 0 <= - (- (3 * kappa * a * m) + 3 * m * a * c).
Proof.
  intros Hm Ha Hk Hc. assert (0 <= m * a) by (apply Rmult_le_pos; assumption). nra.
Qed.

(* the radial jets are derivatives of one another *)
Lemma j_E_derive r : 1 <= r <= 2 -> is_derive j_E r (j_E_r r).
Proof.
  intros Hr. unfold j_E, j_E_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_E_r_derive r : 1 <= r <= 2 -> is_derive j_E_r r (j_E_rr r).
Proof.
  intros Hr. unfold j_E_r, j_E_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_E_u_derive r : 1 <= r <= 2 -> is_derive j_E_u r (j_E_ru r).
Proof.
  intros Hr. unfold j_E_u, j_E_ru. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_e_derive r : 1 <= r <= 2 -> is_derive j_e r (j_e_r r).
Proof.
  intros Hr. unfold j_e, j_e_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_e_r_derive r : 1 <= r <= 2 -> is_derive j_e_r r (j_e_rr r).
Proof.
  intros Hr. unfold j_e_r, j_e_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_e_u_derive r : 1 <= r <= 2 -> is_derive j_e_u r (j_e_ru r).
Proof.
  intros Hr. unfold j_e_u, j_e_ru. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_m_derive r : 1 <= r <= 2 -> is_derive j_m r (j_m_r r).
Proof.
  intros Hr. unfold j_m, j_m_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_m_r_derive r : 1 <= r <= 2 -> is_derive j_m_r r (j_m_rr r).
Proof.
  intros Hr. unfold j_m_r, j_m_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_m_u_derive r : 1 <= r <= 2 -> is_derive j_m_u r (j_m_ru r).
Proof.
  intros Hr. unfold j_m_u, j_m_ru. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_W_derive r : 1 <= r <= 2 -> is_derive j_W r (j_W_r r).
Proof.
  intros Hr. unfold j_W, j_W_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_W_r_derive r : 1 <= r <= 2 -> is_derive j_W_r r (j_W_rr r).
Proof.
  intros Hr. unfold j_W_r, j_W_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_W_u_derive r : 1 <= r <= 2 -> is_derive j_W_u r (j_W_ru r).
Proof.
  intros Hr. unfold j_W_u, j_W_ru. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_U_derive r : 1 <= r <= 2 -> is_derive j_U r (j_U_r r).
Proof.
  intros Hr. unfold j_U, j_U_r. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_U_r_derive r : 1 <= r <= 2 -> is_derive j_U_r r (j_U_rr r).
Proof.
  intros Hr. unfold j_U_r, j_U_rr. auto_derive; [ nonzero | field; nonzero ].
Qed.

Lemma j_U_u_derive r : 1 <= r <= 2 -> is_derive j_U_u r (j_U_ru r).
Proof.
  intros Hr. unfold j_U_u, j_U_ru. auto_derive; [ nonzero | field; nonzero ].
Qed.

End Rocket.
