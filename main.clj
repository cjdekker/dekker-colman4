(def vocabulary '(call me ishmael))
(def theta1 (list (/ 1 2 ) (/ 1 4 ) (/ 1 4 )))
(def theta2 (list (/ 1 4 ) (/ 1 2 ) (/ 1 4 )))
(def thetas (list theta1 theta2))
(def theta-prior (list (/ 1 2) (/ 1 2)))
(defn score-categorical [outcome outcomes params]
  (if (empty? params)
    (/ 1 0)
    (if (= outcome (first outcomes))
      (first params)
      (score-categorical outcome (rest outcomes) (rest params)))))
(defn list-foldr [f base lst]
  (if (empty? lst)
    base
    (f (first lst)
      (list-foldr f base (rest lst)))))
(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))
(defn score-BOW-sentence [sen probabilities]
  (list-foldr
    (fn [word rest-score]
      (+ (log2 (score-categorical word vocabulary probabilities))
        rest-score))
    0
    sen))
(defn score-corpus [corpus probabilities]
  (list-foldr
    (fn [sen rst]
      (+ (score-BOW-sentence sen probabilities) rst))
    0
    corpus))
(defn logsumexp [log-vals]
  (let [mx (apply max log-vals)]
    (+ mx
      (log2
        (apply +
          (map (fn [z] (Math/pow 2 z))
            (map (fn [x] (- x mx)) log-vals)))))))
(def my-corpus '((call me) (call ishmael)))

;PROBLEM 1

(defn theta-corpus-joint [theta corpus theta-probs]
   (+ (score-corpus corpus theta) (log2 (first theta-probs))))

(theta-corpus-joint theta1 my-corpus theta-prior)

;PROBLEM 2

(defn compute-marginal [corpus theta-probs]
 (logsumexp (map + (map (fn [theta] (score-corpus corpus theta)) thetas) (map (fn [y] (log2 y)) theta-probs))))

(compute-marginal my-corpus theta-prior)

;PROBLEM 3

(defn compute-conditional-prob [theta corpus theta-probs]
  (- (theta-corpus-joint theta corpus theta-probs) (compute-marginal my-corpus theta-probs))
  )

;PROBLEM 4

(defn compute-conditional-dist [corpus theta-probs]
  (map (fn [theta] (compute-conditional-prob theta corpus theta-probs))thetas))

;PROBLEM 5

(compute-conditional-dist my-corpus theta-prior)

(map (fn [x] (Math/pow 2 x)) (compute-conditional-dist my-corpus theta-prior))

;According to this information, given our corpus, theta1 is twice as likely as theta2. This makes sense, because it assigns 1/2 instead of 1/4 to 'call and 1/4 instead of 1/2 to 'me, which is more likely because 'call shows up twice in our corpus but 'me only once.

;PROBLEM 6

(defn compute-posterior-predictive [observed-corpus new-corpus theta-probs]
  (let [conditional-dist (map (fn [x] (Math/pow 2 x)) 
  (compute-conditional-dist observed-corpus theta-probs))]
    (compute-marginal new-corpus conditional-dist)))

(compute-posterior-predictive my-corpus my-corpus theta-prior)

;This quantity represents the likelihood of getting our observed outcome using the conditional distribution. This is more accurate than problem 2 because theta1 is weighed more heavily than theta-prior where each theta was treated equally. Theta1 should be weighed more heavily because it assigns 1/2 instead of 1/4 to 'call and 1/4 instead of 1/2 to 'me, which is better because 'call shows up twice in our corpus but 'me only once.

;;;;;;;;;;;;;;
(defn normalize [params]
  (let [sum (apply + params)]
    (map (fn [x] (/ x sum)) params)))
(defn flip [weight]
  (if (< (rand 1) weight)
    true
    false))
(defn sample-categorical [outcomes params]
  (if (flip (first params))
  (first outcomes)
  (sample-categorical (rest outcomes)
    (normalize (rest params)))))
(defn repeat [f n]
  (if (= n 0)
  '()
  (cons (f) (repeat f (- n 1)))))
(defn sample-BOW-sentence [len probabilities]
  (if (= len 0)
    '()
    (cons (sample-categorical vocabulary probabilities)
      (sample-BOW-sentence (- len 1) probabilities))))
;;;;;;;;;;;;;

;PROBLEM 7

(defn sample-BOW-corpus [theta sent-len corpus-len]
  (repeat (fn [] (sample-BOW-sentence sent-len theta)) corpus-len))

;PROBLEM 8

(defn sample-theta-corpus [sent-len corpus-len theta-probs]
  (let [theta (sample-categorical thetas theta-probs)]
    (list theta (sample-BOW-corpus theta sent-len corpus-len))))

;;;;;;;;;;;
(defn get-theta [theta-corpus]
  (first theta-corpus))
(defn get-corpus [theta-corpus]
  (first (rest theta-corpus)))
(defn sample-thetas-corpora [sample-size sent-len corpus-len theta-probs]
  (repeat (fn [] (sample-theta-corpus sent-len corpus-len theta-probs))     sample-size))
;;;;;;;;;;;

;PROBLEM 9

(defn estimate-corpus-marginal [corpus sample-size sent-len
corpus-len theta-probs]
  (let [biglist (map (fn [x] (get-corpus x)) (sample-thetas-corpora sample-size sent-len corpus-len theta-probs))]
  (/ (count (remove false? (map (fn [x] (= x corpus)) biglist)
    )) (count biglist))))

;PROBLEM 10

;(estimate-corpus-marginal my-corpus 50 2 2 theta-prior)
;1/50,1/25,1/50,1/50,0

;(estimate-corpus-marginal my-corpus 1000 2 2 theta-prior)
;3/250,1/250,1/100,11/1000,9/1000

;We get fairly similar answers when using a sample-size of 50 vs 1000, but with a little more variance when using the smaller sample size. This makes sense, because as the sample size approaches infitity, our approximation approaches the exact value. These calls give us average values of a little over .01, whereas our exact value from problem 2 is equal to about .013 when exponentiated. Thus our approximations are very close to the exact marginal likelihood.

;;;;;;;;;;
(defn get-count [obs observation-list count]
(if (empty? observation-list)
count
(if (= obs (first observation-list))
(get-count obs (rest observation-list) (+ 1 count))
(get-count obs (rest observation-list) count))))
(defn get-counts [outcomes observation-list]
(let [count-obs (fn [obs] (get-count obs observation-list 0))]
(map count-obs outcomes)))
;;;;;;;;;;

;PROBLEM 11

(defn hp [l1 l2]
  (if (empty? l2)
    '()
    (if (= (first l2) true)
      (cons (first l1) (hp (rest l1) (rest l2)))
      (hp (rest l1) (rest l2))
  )))

(defn rejection-sampler [theta observed-corpus sample-size sent-len corpus-len theta-probs]
      (let [biglist (map (fn [x] (get-corpus x)) (sample-thetas-corpora sample-size sent-len corpus-len theta-probs))] 
        (let [thlist (map (fn [x] (get-theta x)) (sample-thetas-corpora sample-size sent-len corpus-len theta-probs))]
          (let [blist (map (fn [x] (= x observed-corpus)) biglist)]
            (let [xlist (hp thlist blist)]          
                (/ (get-count theta xlist 0) (count xlist))
        )))))

;PROBLEM 12

;After running it a few times with sample size 100, I observed a few 0s, a few 1s, a few NaNs, a few 1/2s and little else. This makes sense, because on average only 1.3 times the corpus will match any of the sample corpora for every 100 samples, so our denominator will seldom be much larger than 2. It takes a sample size of at least a few thousand to stabilize, which should be expected, because only a little more than 1% of our samples even end up being considered.