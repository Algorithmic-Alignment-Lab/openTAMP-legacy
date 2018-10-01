(define (domain sorting_domain)
    (:requirements :equality)
    (:predicates (ClothAtTarget ?cloth - Cloth ?target - Target)
                 (ClothInGripperLeft ?cloth - Cloth)
                 (ClothInGripperRight ?cloth - Cloth)
                 (ClothInReach ?cloth - Cloth)
                 (TargetIsOnLeft ?target - Target)
                 (TargetIsOnRight ?target - Target)
                 (ClothIsOnLeft ?cloth - Cloth)
                 (ClothIsOnRight ?cloth - Cloth)
    )

    (:types Cloth Target)

    (:action grasp_left
        :parameters (?cloth - Cloth)
        :precondition (and (not (ClothInGripperRight ?cloth))
                           (forall (?c - Cloth) (not (ClothInGripperLeft ?c)))
                           (ClothIsOnLeft ?cloth))
        :effect (ClothInGripperLeft ?cloth)
    )

    (:action putdown_left
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (ClothInGripperLeft ?cloth)
                           (forall (?c - Cloth) (not (ClothAtTarget ?c ?target)))
                           (TargetIsOnLeft ?target))
        :effect (and (not (ClothInGripperLeft ?cloth))
                     (ClothAtTarget ?cloth ?target)
                     (ClothIsOnLeft ?cloth)
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (ClothAtTarget ?cloth ?t))))
                     (when (TargetIsOnRight ?target) (ClothIsOnRight ?cloth)))
    )

    (:action grasp_right
        :parameters (?cloth - Cloth)
        :precondition (and (not (ClothInGripperLeft ?cloth))
                           (forall (?c - Cloth) (not (ClothInGripperRight ?c)))
                           (ClothIsOnRight ?cloth))
        :effect (ClothInGripperRight ?cloth)
    )

    (:action putdown_right
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (ClothInGripperRight ?cloth)
                           (forall (?c - Cloth) (not (ClothAtTarget ?c ?target)))
                           (TargetIsOnRight ?target))
        :effect (and (not (ClothInGripperRight ?cloth))
                     (ClothAtTarget ?cloth ?target)
                     (ClothIsOnRight ?cloth)
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (ClothAtTarget ?cloth ?t))))
                     (when (TargetIsOnLeft ?target) (ClothIsOnLeft ?cloth)))
    )
)