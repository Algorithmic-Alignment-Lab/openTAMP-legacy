(define (domain sorting_domain)
    (:requirements :equality)
    (:predicates (ClothAtTarget ?cloth - Cloth ?target - Target)
                 (ClothInGripperLeft ?cloth - Cloth)
                 (ClothInGripperRight ?cloth - Cloth)
                 (ClothInReachLeft ?cloth - Cloth)
                 (ClothInReachRight ?cloth - Cloth)
                 (TargetInReachLeft ?cloth - Cloth)
                 (TargetInReachRight ?cloth - Cloth)
    )

    (:types Cloth Target)

    (:action grasp_left
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (forall (?c - Cloth) (not (ClothInGripperLeft ?c)))
                           (not (ClothInGripperRight ?cloth))
                           (ClothInReachLeft ?cloth))
        :effect (and (forall (?t - Target) (not (ClothAtTarget ?cloth ?t)))
                     (ClothInGripperLeft ?cloth))
    )
    (:action putdown_left
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (ClothInGripperLeft ?cloth)
                           (TargetInReachLeft ?target))
        :effect (and (not (ClothInGripperLeft ?cloth))
                     (when (TargetInReachLeft ?target) (ClothInReachLeft ?cloth))
                     (when (TargetInReachRight ?target) (ClothInReachRight ?cloth))
                     (when (not (TargetInReachLeft ?target)) (not (ClothInReachLeft ?cloth)))
                     (when (not (TargetInReachRight ?target)) (not (ClothInReachRight ?cloth))))
    )

    (:action grasp_right
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (forall (?c - Cloth) (not (ClothInGripperRight ?c)))
                           (not (ClothInGripperLeft ?cloth))
                           (ClothInReachRight ?cloth))
        :effect (and (forall (?t - Target) (not (ClothAtTarget ?cloth ?t)))
                     (ClothInGripperRight ?cloth))
    )
    (:action putdown_right
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (ClothInGripperRight ?cloth)
                           (TargetInReachRight ?target))
        :effect (and (not (ClothInGripperRight ?cloth))
                     (when (TargetInReachRight ?target) (ClothInReachRight ?cloth))
                     (when (TargetInReachLeft ?target) (ClothInReachLeft ?cloth))
                     (when (not (TargetInReachLeft ?target)) (not (ClothInReachLeft ?cloth)))
                     (when (not (TargetInReachRight ?target)) (not (ClothInReachRight ?cloth))))
    )

)
