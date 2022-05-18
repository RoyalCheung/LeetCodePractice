import javax.security.auth.PrivateCredentialPermission;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class Heap {
    //215
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        for (int i = 0; i < nums.length;i++){
            queue.add(nums[i]);
        }
        int result = 0;
        for (int i = 0; i < k; i++){
            result = queue.remove();
        }
        return result;
    }
    //218
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> result = new ArrayList<>();
        List<List<Integer>> height = new ArrayList<>();
        for (int[] building : buildings){
            List<Integer> temp = new ArrayList<>();
            temp.add(building[0]);
            temp.add(-building[2]);
            height.add(temp);
            temp = new ArrayList<>();
            temp.add(building[1]);
            temp.add(building[2]);
            height.add(temp);
        }
        Collections.sort(height, (a, b) -> {
            if (a.get(0) != b.get(0)){
                return a.get(0) - b.get(0);}
            return a.get(1) - b.get(1);
        });
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> (b - a));
        int prev = 0;
        queue.add(prev);
        int sig  = -1;
        int cur = 0;
        for (List<Integer> h : height){
            if (sig != -1){
                if (h.get(0) != sig){
                    List<Integer> temp = new ArrayList<>();
                    temp.add(sig);
                    temp.add(cur);
                    result.add(temp);
                    prev = cur;
                }
                sig = -1;
            }
            if (h.get(1) < 0){
                queue.add(-h.get(1));
                cur = queue.peek();
                if (cur != prev){
                    List<Integer> temp = new ArrayList<>();
                    temp.add(h.get(0));
                    temp.add(cur);
                    result.add(temp);
                    prev = cur;
                }
            }else{
                queue.remove(h.get(1));
                cur = queue.peek();
                if (cur != prev) {
                    sig = h.get(0);
                }
            }
        }
        if (sig != -1){
            List<Integer> temp = new ArrayList<>();
            temp.add(sig);
            temp.add(cur);
            result.add(temp);
            prev = cur;
        }

        return result;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
            if (nums.length ==0 || k <=0) return new int[0];
            int[] result = new int[nums.length-k+1];
            Deque<Integer> queue = new ArrayDeque<>();
            int resultI = 0;
            for (int i = 0; i < nums.length; i++){
                while (!queue.isEmpty() && queue.peek() < i -k + 1){
                    queue.remove();
                }
                while (!queue.isEmpty() && nums[queue.peekLast()] < nums[i]){
                    queue.removeLast();
                }
                queue.add(i);
                if (i >= k - 1){
                    result[resultI++] = nums[queue.peek()];
                }
            }
            return result;
    }
    // 264
    public int nthUglyNumber(int n) {
        int[] nums = new int[n];
        nums[0] = 1;
        int index2 = 0;
        int index3 = 0;
        int index5 = 0;
        int factor2 = 2;
        int factor3 = 3;
        int factor5 = 5;
        for (int i = 1; i < n; i++){
            int temp = Math.min(factor2, Math.min(factor3, factor5));
            nums[i] = temp;
            if (factor2 == temp){
                factor2 = 2 * nums[++index2];
            }
            if (factor3 == temp){
                factor3 = 3 * nums[++index3];
            }
            if (factor5 == temp){
                factor5 = 5* nums[++index5];
            }
        }
        return nums[n-1];
    }
    // 295
    class MedianFinder {
        PriorityQueue<Integer> large = new PriorityQueue<>();
        PriorityQueue<Integer> small = new PriorityQueue<>((a, b) -> (b-a));
        public MedianFinder() {
        }

        public void addNum(int num) {
            large.add(num);
            small.add(large.remove());
            if (small.size() > large.size()){
                large.add(small.remove());
            }
        }

        public double findMedian() {
            Double result = large.size() > small.size() ? large.peek() : Double.valueOf(large.peek()+small.peek())/2;
            return result;
        }
    }

    //313
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] nums = new int[n];
        nums[0] = 1;
        int[] indexes = new int[primes.length];
        PriorityQueue<List<Integer>> factors = new PriorityQueue<>((a, b) -> (a.get(0) - b.get(0)));
        for (int i = 0; i < primes.length; i++) {
            List<Integer> temp = new ArrayList<>();
            temp.add(primes[i]);
            temp.add(i);
            factors.add(temp);
        }
        int i = 1;
        while (i < n){
            List<Integer> temp = factors.remove();
            if (temp.get(0) != nums[i-1]) {
                nums[i] = temp.get(0);
                i++;
            }
            int number = temp.get(1);
            int newFactor = primes[temp.get(1)] * nums[++indexes[temp.get(1)]];
            temp = new ArrayList<>();
            temp.add(newFactor);
            temp.add(number);
            factors.add(temp);
        }
        return nums[n-1];
    }
    //347
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        PriorityQueue<List<Integer>> queue = new PriorityQueue<>((a, b) -> (b.get(1) - a.get(1)));
        for (int i = 0; i < nums.length; i++){
            map.putIfAbsent(nums[i], 0);
            int temp = map.get(nums[i]);
            map.put(nums[i], ++temp);
        }
        for (Integer key : map.keySet()){
            List<Integer> temp = new ArrayList<>();
            temp.add(key);
            temp.add(map.get(key));
            queue.add(temp);
        }
        int[] result = new int[k];
        for (int i = 0; i < k; i++){
            List<Integer> l = queue.remove();
            result[i] = l.get(0);
        }
        return result;
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int[] indexes = new int[nums1.length];
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0 || k == 0) return result;
        PriorityQueue<List<Integer>> queue = new PriorityQueue<>((a, b) -> (a.get(0) + a.get(1) - b.get(0) - b.get(1)));
        int i = 0;
        for (Integer num : nums1){
            if (i > k) break;
            List<Integer> temp = new ArrayList<>();
            temp.add(num);
            temp.add(nums2[0]);
            temp.add(0);
            queue.add(temp);
            i++;
        }
        i = 0;
        while (i < k && !queue.isEmpty()){
            List<Integer> temp = queue.remove();
            int num1 = temp.get(0);
            int num2 = temp.get(1);
            int num3 = temp.get(2);
            List<Integer> temp2 = new ArrayList<>();
            temp2.add(num1);
            temp2.add(num2);
            result.add(temp2);
            if (num3 < nums2.length-1){
                temp2 = new ArrayList<>();
                temp2.add(num1);
                temp2.add(nums2[++num3]);
                temp2.add(num3);
                queue.add(temp2);
            }
            i++;
        }
        return result;
    }
    //378
    public int kthSmallest(int[][] matrix, int k) {
        int result = matrix[0][0];
        PriorityQueue<Tuple> queue = new PriorityQueue<>((a, b) -> (a.val - b.val));
        for (int i = 0; i < matrix.length; i++){
            queue.add(new Tuple(0, i, matrix[0][i]));
        }
        int i = 0;
        while (i < k && !queue.isEmpty()){
            Tuple temp = queue.remove();
            if (temp.x < matrix.length-1) {
                queue.add(new Tuple(temp.x + 1, temp.y, matrix[temp.x + 1][temp.y]));
            }
            result = temp.val;
            i++;
        }

        return result;
    }

    class Tuple {
        public int x;
        public int y;
        public int val;
        public Tuple(int x, int y, int val){
            this.x = x;
            this.y = y;
            this.val = val;
        }
    }

    public String frequencySort(String s) {
        StringBuilder sb = new StringBuilder();
        HashMap<Character, Integer> map = new HashMap<>();
        PriorityQueue<Character> queue = new PriorityQueue<>((a, b) -> (map.get(b) - map.get(a)));
        for (char c : s.toCharArray()){
            int freq = map.getOrDefault(c, 0) + 1;
            map.put(c, freq);
        }
        for (Character c : map.keySet()){
            queue.add(c);
        }
        while (!queue.isEmpty()){
            char c = queue.remove();
            int freq = map.get(c);
            for (int i = 0; i< freq; i++){
                sb.append(c);
            }
        }

        return sb.toString();
    }
    public double[] medianSlidingWindow(int[] nums, int k) {
        PriorityQueue<Double> larger = new PriorityQueue<>();
        PriorityQueue<Double> smaller = new PriorityQueue<>((a,b) -> (int) (b - a));

        int rcur = 0;
        double[] result = new double[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++){
            larger.add(Double.valueOf(nums[i]));
            smaller.add(larger.remove());
            if (smaller.size() > larger.size()){
                larger.add(smaller.remove());
            }
            if (i >= k-1){
                result[rcur] = larger.size() > smaller.size() ? larger.peek() :
                        (larger.peek() + smaller.peek())/2;
                if (larger.peek() <= nums[rcur]){
                    larger.remove(Double.valueOf(nums[rcur]));
                    if (smaller.size() > larger.size()){
                        larger.add(smaller.remove());
                    }
                }else{
                    smaller.remove(Double.valueOf(nums[rcur]));
                    if (smaller.size() < larger.size()-1){
                        smaller.add(larger.remove());
                    }
                }
                rcur++;
            }
        }
        return result;
    }

    //502
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        HashMap<Integer, Integer[]> map = new HashMap<>();
        PriorityQueue<Integer> queuePro = new PriorityQueue<>((a, b) -> {
            return profits[b] - profits[a];
        });
        PriorityQueue<Integer> queueCap = new PriorityQueue<>((a, b) -> {
            return capital[a] - capital[b];
        });
        for (int i = 0; i < profits.length; i++) {
            queueCap.add(i);
        }
        int sum = w;
        for (int i = 0; i < k; i++) {
            while (!queueCap.isEmpty() && capital[queueCap.peek()] <= sum) {
                int point = queueCap.remove();
                queuePro.add(point);
            }
            if (queuePro.isEmpty()) {
                break;
            } else {
                int point = queuePro.remove();
                sum += profits[point];
            }
        }


        return sum;
    }
    //621
    public int leastInterval(char[] tasks, int n) {
        if (n == 0) return tasks.length;
        HashMap<Character, Integer> map = new HashMap<>();
        int[] count = new int[26];
        int max = 0;
        int maxCount = 0;
        for (char c : tasks){
            count[c-'A']++;
            if (max == count[c-'A']){
                max = count[c-'A'];
                maxCount++;
            }else if (max < count[c-'A']){
                max = count[c-'A'];
                maxCount = 1;
            }
        }
        int temp = Math.max(0, (n - maxCount + 1) * (max-1) - (tasks.length - maxCount * max)) + tasks.length;

        return temp;
    }

    //630
    public int scheduleCourse(int[][] courses) {
        PriorityQueue<int[]> queueLast = new PriorityQueue<>((a, b) -> {
            return a[1] - b[1];
        });
        PriorityQueue<Integer> queueDure = new PriorityQueue<>((a, b) ->{
            return b - a;
        });
        for (int i = 0; i < courses.length; i++){
            queueLast.add(courses[i]);
        }
        int time = 0;
        while (!queueLast.isEmpty()){
            int[] temp = queueLast.remove();
            queueDure.add(temp[0]);
            time+= temp[0];
            if (time > temp[1]){
                time -= queueDure.remove();
            }
        }
        return queueDure.size();
    }

    //632
    public int[] smallestRange(List<List<Integer>> nums) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> {
            return a[1] - b[1];
        });
        int max = 0;
        for (int i = 0; i < nums.size(); i++){
            queue.add(new int[]{i, nums.get(i).get(0), 0});
            if (nums.get(i).get(0) > max){
                max = nums.get(i).get(0);
            }
        }
        int start = 0, end = 0, range = Integer.MAX_VALUE;

        while (queue.size() == nums.size()){
            int[] prev = queue.remove();
            if ((max - prev[1]) < range){
                start = prev[1];
                end = max;
                range = max - prev[1];
            }
            int i = prev[0];
            int j = prev[2] + 1;
            if (j < nums.get(i).size()){
                int temp = nums.get(i).get(j);
                if (temp > max) max = temp;
                queue.add(new int[]{i, temp, j});
            }
        }
        return new int[]{end, start};
    }
    //658
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> result = new ArrayList<>();
        PriorityQueue<Integer> smaller = new PriorityQueue<>((a,b) -> (b-a));
        PriorityQueue<Integer> larger = new PriorityQueue<>();
        for (int i = 0; i < arr.length; i++){
            if (arr[i] >= x){
                larger.add(arr[i]);
            }else{
                smaller.add(arr[i]);
            }
        }
        while (k > 0){
            if (smaller.isEmpty() || larger.peek() - x < x - smaller.peek()){
                result.add(larger.peek());
                larger.remove();
            }else if(larger.isEmpty()) {
                result.add(0, smaller.peek());
                smaller.remove();
            }
            else if (larger.peek() - x < x - smaller.peek()){
                result.add(larger.peek());
                larger.remove();
            }
            else {
                result.add(0, smaller.peek());
                smaller.remove();
            }
            k--;
        }
        return result;
    }

    //659
    public boolean isPossible(int[] nums) {
        int pre = Integer.MAX_VALUE, p1 = 0, p2 = 0, p3 = 0, c1 = 0, c2 = 0, c3 = 0, cnt = 0, cur = 0;
        for (int i = 0; i < nums.length; pre = cur, p1 = c1, p2 = c2, p3 = c3){
            for (cur = nums[i], cnt = 0; i < nums.length && nums[i] == cur; i++){
                cnt++;
            }
            if (cur != pre + 1){
                if (p1 != 0 || p2 != 0){
                    return false;
                }
                c1 = cnt;
                c2 = 0;
                c3 = 0;
            }else{
                if (cnt < (p1 + p2)){
                    return false;
                }
                c1 = Math.max(0, cnt - (p1+p2+p3));
                c2 = p1;
                c3 = p2 + Math.min(p3, cnt - (p1+p2));
            }
        }
        return (p1 == 0 && p2 == 0);
    }
    // 675
    public int cutOffTree(List<List<Integer>> forest) {
        if (forest == null || forest.size() == 0) return 0;
        int[][] dir = {{0,1}, {0,-1}, {1,0}, {-1,0}};
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> (a[2] - b[2]));
        int m = forest.size(), n = forest.get(0).size();
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                if (forest.get(i).get(j) > 1) {
                    queue.add(new int[]{i, j, forest.get(i).get(j)});
                }
            }
        }
        int[] start = new int[2];
        int sum = 0;
        while (!queue.isEmpty()){
            int[] cur = queue.remove();
            int step = cutOffTreeBFS(forest, start, cur, dir);
            if (step == -1) return -1;
            sum += step;
            start = new int[2];
            System.arraycopy(cur, 0, start, 0, 2);
        }
        return sum;
    }

    public int cutOffTreeBFS(List<List<Integer>> forest, int[] start, int end[], int[][] dir){
        Queue<int[]> queue = new ArrayDeque<>();
        queue.add(start);
        int step = 0;
        int m = forest.size(), n = forest.get(0).size();
        boolean[][] visited = new boolean[m][n];
        visited[start[0]][start[1]] = true;
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0; i < size; i++){
                int[] cur = queue.remove();
                if (cur[0] == end[0] && cur[1] == end[1]) return step;
                for (int[] d : dir) {
                    int x = cur[0] + d[0];
                    int y = cur[1] + d[1];
                    if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && forest.get(x).get(y) != 0) {
                        queue.add(new int[]{x, y});
                        visited[x][y] = true;
                    }
                }
            }
            step++;
        }
        return -1;
    }
    //692
    public List<String> topKFrequent(String[] words, int k) {
        HashMap<String, Integer> map = new HashMap<>();
        for (String word : words){
            int temp  = map.getOrDefault(word, 0) + 1;
            map.put(word, temp);
        }
        PriorityQueue<String> queue = new PriorityQueue<>((a,b) -> {
            if (map.get(a) == map.get(b)){
                return a.compareTo(b);
            }
            return map.get(b) - map.get(a);
        });
        for (String key : map.keySet()){
            queue.add(key);
        }
        List<String> result = new ArrayList<>();
        for (int i = 0; i < k; i++){
            String temp = queue.remove();
            result.add(temp);
        }
        return result;
    }
    // 703
    class KthLargest {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> {
            return b - a;
        });
        PriorityQueue<Integer> larger = new PriorityQueue<>();
        int k = 0;
        public KthLargest(int k, int[] nums) {
            this.k = k;
            for (int i : nums){
                queue.add(i);
            }
            for (int i = 0; i < k && !queue.isEmpty(); i++){
                larger.add(queue.remove());
            }
        }

        public int add(int val) {
            if (larger.isEmpty() || larger.size() < k){
                larger.add(val);
            }
            else if (!larger.isEmpty() && val > larger.peek()){
                larger.remove();
                larger.add(val);
            }
            return larger.peek();
        }
    }
    //767
    public String reorganizeString(String s) {
        if (s.length() == 1) return s;
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()){
            int temp = map.getOrDefault(c, 0) + 1;
            map.put(c, temp);
        }
        PriorityQueue<Character> queue = new PriorityQueue<>((a,b) -> {
            return map.get(b) - map.get(a);
        });
        for (Character c : map.keySet()){
            queue.add(c);
        }
        char[] result = new char[s.length()];
        char temp = queue.remove();
        if (map.get(temp) < s.length()/2-1) return "";
        int j = 0;
        for (int i = 0; i < map.get(temp); i++){
            result[j] = temp;
            j+=2;
        }
        while (!queue.isEmpty()){
            temp = queue.remove();
            for (int i = 0; i < map.get(temp); i++){
                if (j > result.length-1) j = 1;
                result[j] = temp;
                j += 2;
            }
        }

        return String.valueOf(result);
    }
    //778
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        int[][] dir = {{0,1}, {0,-1}, {1,0}, {-1,0}};
        int time = grid[0][0];
        while (true){
            time++;
            boolean[][] checked = new boolean[n][n];
            checked[0][0] = true;
            boolean tempB = swimInWaterDFS(grid, new int[]{0,0}, checked, dir, time);
            if (tempB) {
                break;
            }
        }
        return time;
    }
    public boolean swimInWaterDFS(int[][] grid, int[] start, boolean[][] checked, int[][] dir, int time){
        if (start[0] == grid.length-1 && start[1] == grid.length-1) return true;
        checked[start[0]][start[1]] = true;
        for (int[] d : dir){
            int[] temp = {start[0] + d[0], start[1] + d[1]};
            if (temp[0] >= 0 && temp[0] < grid.length && temp[1] >= 0 && temp[1] < grid.length && grid[temp[0]][temp[1]] <= time
                    && !checked[temp[0]][temp[1]]){
                boolean tempB = swimInWaterDFS(grid, temp, checked, dir, time);
                if (tempB) return true;
            }
        }
        return false;
    }
    //786
    public int[] kthSmallestPrimeFraction(int[] arr, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> {
            Double temp = (double) a[0]/a[1] - (double) b[0]/b[1];
            if (temp < 0) return -1;
            return 1;
        });
        for (int i = 0; i < arr.length-1; i++){
            for (int j = i +1; j < arr.length; j++){
                pq.add(new int[]{arr[i], arr[j]});
            }
        }
        int[] temp = new int[2];
        while (k > 0) {
            temp = pq.remove();
            k--;
        }
        return temp;
    }
    //857
    public double mincostToHireWorkers(int[] quality, int[] wage, int k) {
        double sum = 0;
        double result = Double.MAX_VALUE;
        PriorityQueue<Double> pq = new PriorityQueue<>((a,b) -> {
            return b.compareTo(a);
        });
        PriorityQueue<Integer> pq2 = new PriorityQueue<>((a,b) -> {
            Double temp1 = (double) wage[a] / quality[a];
            Double temp2 = (double) wage[b] / quality[b];
            return temp1.compareTo(temp2);
        });
        for (int i = 0; i < quality.length; i++){
            pq2.add(i);
        }
        while (!pq2.isEmpty()){
            int temp = pq2.remove();
            double ratio = (double) wage[temp] / quality[temp];
            sum += (double) quality[temp];
            pq.add((double) quality[temp]);
            if (pq.size() > k) {
                sum -= pq.remove();
            }
            if (pq.size() == k){
                result = Math.min(result, ratio*sum);
            }
        }
        return result;
    }
    //862
    public int shortestSubarray(int[] nums, int k) {
        double[] sums = new double[nums.length+1];
        for (int i = 0; i < nums.length; i++){
            sums[i+1] = sums[i] + (double) nums[i];
        }
        Deque<Integer> deque = new ArrayDeque<>();
        int result = sums.length;
        for (int i = 0; i < sums.length; i++){
            while (!deque.isEmpty() && (sums[i] - sums[deque.peekFirst()]) >= k){
                result = Math.min(result, i - deque.removeFirst());
            }
            while (!deque.isEmpty() && (sums[i] <= sums[deque.peekLast()])){
                deque.removeLast();
            }
            deque.addLast(i);
        }
        return result <= nums.length ? result : -1;
    }
    //871
//    public int minRefuelStops(int target, int startFuel, int[][] stations) {
//        double[] dp = new double[stations.length+1];
//        dp[0] = startFuel;
//        for (int i = 0; i < stations.length; i++){
//            for (int j = i; j >= 0 && dp[j] >= stations[i][0];j--){
//                dp[j+1] = Math.max(dp[j+1], dp[j]+stations[i][1]);
//            }
//        }
//        for (int i = 0; i < dp.length; i++){
//            if(dp[i] >= target){
//                return i;
//            }
//        }
//        return -1;
//    }
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        PriorityQueue<Double> pq = new PriorityQueue<>((a,b) -> {
            return b.compareTo(a);
        });
        int res = 0;
        double sum = startFuel;
        int i = 0;
        for (res =0; sum < target;res++){
            while (i < stations.length && sum >= stations[i][0]){
                pq.add((double)stations[i++][1]);
            }
            if (pq.isEmpty()) return -1;
            sum += pq.remove();
        }
        return res;
    }

    //882
    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        HashMap<Integer, HashMap<Integer, Integer>> adj = new HashMap<>();
        for (int[] edge : edges){
            adj.computeIfAbsent(edge[0],k -> new HashMap<>()).put(edge[1], edge[2]);
            adj.computeIfAbsent(edge[1],k -> new HashMap<>()).put(edge[0], edge[2]);

        }
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> {
            return b[0] - a[0];
        });
        HashMap<Integer, Integer> seen = new HashMap<>();
        pq.add(new int[]{maxMoves, 0});
        while (!pq.isEmpty()){
            int[] tempCur = pq.remove();
            if (!seen.containsKey(tempCur[1])) {
                seen.put(tempCur[1], tempCur[0]);
                if (adj.containsKey(tempCur[1])) {
                    for (int j : adj.get(tempCur[1]).keySet()) {
                        if (!seen.containsKey(j) && adj.get(tempCur[1]).get(j) + 1 <= tempCur[0]) {
                            pq.add(new int[]{tempCur[0] - adj.get(tempCur[1]).get(j) - 1, j});
                        }
                    }
                }
            }
        }
        int result = seen.size();
        for (int[] edge : edges){
            int a = seen.getOrDefault(edge[0], 0);
            int b = seen.getOrDefault(edge[1], 0);
            result += Math.min(a+b, edge[2]);
        }

        return result;
    }
    //912
    public int[] sortArray(int[] nums) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i = 0; i < nums.length; i++){
            pq.add(nums[i]);
        }
        int[] result = new int[nums.length];
        int i = 0;
        while (!pq.isEmpty()){
            result[i++] = pq.remove();
        }
        return result;
    }
    //973
    public int[][] kClosest(int[][] points, int k) {
        int[][] result = new int[k][2];
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> {
            return a[0]*a[0]+a[1]*a[1] - (b[0]*b[0]+b[1]*b[1]);
        });
        for (int[] point : points){
            pq.add(point);
        }
        int i = 0;
        while (!pq.isEmpty() && i < k){
            result[i++] = pq.remove();
        }
        return result;
    }

    //1046
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a,b) -> {
            return b-a;
        });
        for (int stone : stones){
            pq.add(stone);
        }
        while (pq.size() > 1){
            pq.add(pq.remove() - pq.remove());
        }
        return pq.peek();
    }
    //1054
    public int[] rearrangeBarcodes(int[] barcodes) {
        int[] result = new int[barcodes.length];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int barcode : barcodes){
            int temp = map.getOrDefault(barcode, 0) + 1;
            map.put(barcode, temp);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((a,b) -> {
            return map.get(b) - map.get(a);
        });
        int i = 0;
        for (int key : map.keySet()){
            pq.add(key);
        }
        while (!pq.isEmpty()){
            int temp = pq.remove();
            for (int j = 0; j < map.get(temp); j++, i+=2){
                if (i > result.length-1) i = 1;
                result[i] = temp;
            }

        }
        return result;
    }
    //1094
    public boolean carPooling(int[][] trips, int capacity) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> {
            if (a[0] == b[0]){
                return a[1] - b[1];
            }
            return a[0] - b[0];
        });
        for (int[] trip : trips){
            pq.add(new int[]{trip[1], trip[0]});
            pq.add(new int[]{trip[2], -trip[0]});
        }
        int sum = 0;
        while (!pq.isEmpty()){
            int[] trip = pq.remove();
            sum += trip[1];
            if (sum > capacity) return false;
        }
        return true;
    }

    static class DinnerPlates {
        int capacity;
        double right;
        double left;
        int count = 0;
        List<Stack<Integer>> store = new ArrayList<>();
//        PriorityQueue<Integer> pq = new PriorityQueue<>((a,b) -> {
//            if (store.get(a).size() < capacity && store.get(b).size() < capacity){
//                return a - b;
//            }
//            return store.get(a).size() - store.get(b).size();
//        });
//        PriorityQueue<Integer> pqright = new PriorityQueue<>((a,b) -> {
//            if (store.get(a).size() == 0 && store.get(b).size() != 0){
//                return 1;
//            }
//            if (store.get(b).size() == 0 && store.get(a).size() != 0){
//                return -1;
//            }
//            return b - a;
//        });
        public DinnerPlates(int capacity) {
            this.capacity = capacity;
            store.add(new Stack<>());
            left = 0;
            right = 0;
        }

        public void push(int val) {
            while (left < store.size() && store.get((int)left).size() == capacity){
                left++;
            }
            if (left == store.size()){
                store.add(new Stack<>());
            }
            store.get((int)left).add(val);
            right = Math.max(right, left);
            count++;
        }

        public int pop() {
            if (count == 0) return -1;
            while (right >= 0 && store.get((int)right).size() == 0){
                right--;
            }
            int result = store.get((int)right).pop();
            left = Math.min(left, right);
            count--;
            return result;
        }

        public int popAtStack(int index) {
            if (index >= store.size()||store.get(index).size() == 0){
                return -1;
            }
            int result = store.get(index).pop();
            left = Math.min(left, index);
            count--;
            return result;
        }
    }
    public void DinnerPlatesInPuT() throws IOException {
        try {
            URL url = new URL("https://leetcode.com/submissions/detail/697231789/testcase/");
            Scanner s = new Scanner(url.openStream());

        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }
    //3
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        int max = 0, j = 0;
        for (int i = 0; i < s.length(); i++){
            if (map.containsKey(s.charAt(i))){
                j = Math.max(j, map.get(s.charAt(i))+1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i-j+1);
        }
        return max;
    }
    //5
    public String longestPalindrome(String s) {
        String result = null;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = s.length()-1; i >= 0; i--){
            for (int j = i; j < s.length(); j++){
                if (s.charAt(i) == s.charAt(j) &&( (j-i) < 3 || dp[i+1][j-1])){
                    dp[i][j] = true;
                }
                if (result == null || ((j - i + 1 > result.length()) && dp[i][j])){
                    result = s.substring(i, j+1);
                }
            }
        }
        return result;
    }
    //6
    public String convert(String s, int numRows) {
        StringBuffer[] sb = new StringBuffer[numRows];
        for (int i = 0; i < numRows; i++){
            sb[i] = new StringBuffer();
        }
        int i = 0;
        while (i < s.length()){
            for (int j = 0; j < numRows && i < s.length(); j++){
                sb[j].append(s.charAt(i++));
            }
            for (int j = numRows - 2; j >=1 && i < s.length(); j--){
                sb[j].append(s.charAt(i++));
            }
        }
        for (int j = 1; j < numRows; j++){
            sb[0].append(sb[j]);
        }
        return sb[0].toString();
    }

    //8
    public int myAtoi(String s) {
        if (s.equals("")) return 0;
        int i = 0, sign = 1, sum = 0;
        while (i < s.length() && s.charAt(i) == ' '){
            i++;
        }
        if (i < s.length() && (s.charAt(i) == '-' || s.charAt(i) == '+')){
            sign = 1 - (s.charAt(i++) -'+');
        }
        while (i < s.length()  && s.charAt(i) <= '9' && s.charAt(i) >= '0' ){
            if (sum > Integer.MAX_VALUE /10 || (sum == Integer.MAX_VALUE/10 && s.charAt(i) > '7')){
                if (sign == 1) {
                    return Integer.MAX_VALUE;
                }else{
                    return Integer.MIN_VALUE;
                }
            }
            sum = sum * 10 +( s.charAt(i++) - '0');
        }
        return sign * sum;
    }
    //10
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) return false;
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        dp[0][0] = true;
        for (int j = 0; j < p.length(); j++){
            if (p.charAt(j) == '*' && dp[0][j-1]){
                dp[0][j+1] = true;
            }
        }
        for (int i = 0; i < s.length(); i++){
            for (int j = 0; j < p.length(); j++){
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.'){
                    dp[i+1][j+1] = dp[i][j];
                }
                if (p.charAt(j) == '*'){
                    if (p.charAt(j-1) != s.charAt(i) && p.charAt(j-1) != '.'){
                        dp[i+1][j+1] = dp[i+1][j-1];
                    }else{
                        dp[i+1][j+1] = (dp[i][j+1] || dp[i][j-1] || dp[i+1][j-1]);
                    }
                }
            }
        }
        return dp[s.length()][p.length()];
    }
    //12
    public String intToRoman(int num) {
        String[] M = {"", "M", "MM", "MM"};
        String[] C = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] X = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] I = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};

        return M[num/1000] + C[num%1000/100] + X[num%100/10] + I[num%10];
    }

    //13
    public int romanToInt(String s) {

        char[] c = s.toCharArray();
        HashMap<Character, Integer> map = new HashMap<>();
        map.put('M', 1000);
        map.put('D', 500);
        map.put('C', 100);
        map.put('L', 50);
        map.put('X', 10);
        map.put('V', 5);
        map.put('I', 1);
        List<Integer> temp = new ArrayList<>();
        for (int i = 0; i < c.length; i++){
            temp.add(map.get(s.charAt(i)));
        }
        int sum = temp.get(0);
        for (int i = 1; i < temp.size(); i++){
            sum += temp.get(i);
            if (temp.get(i)>temp.get(i-1)){
                sum -= 2 * temp.get(i-1);
            }
        }
        return sum;
    }
    //14
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        String pre = strs[0];
        for (int i = 1; i < strs.length; i++){
            while (strs[i].indexOf(pre) != 0){
                pre = pre.substring(0, pre.length()-1);
            }
        }
        return pre;
    }
    //17
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if (digits.isEmpty()) return result;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        result.add("");
        for (int i = 0; i < digits.length(); i++){
            int x = digits.charAt(i) - '0';
            while (result.get(0).length() == i){
                String temp = result.remove(0);
                for (char c : mapping[i].toCharArray()){
                    result.add(temp+c);
                }
            }
        }
        return result;
    }
    //20
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i) == '{'){
                stack.push('}');
            }else if (s.charAt(i) == '('){
                stack.push(')');
            }else if (s.charAt(i) == '['){
                stack.push(']');
            }else if (stack.isEmpty() || stack.pop() != s.charAt(i)){
                return false;
            }
        }
        return stack.isEmpty();
    }
    //22
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesisHelper(result, 0, 0 , n, "");
        return result;
    }
    public void generateParenthesisHelper(List<String> result, int open, int close, int n, String s) {
        if (s.length() == n *2){
            result.add(s);
        }
        if (open < n){
            generateParenthesisHelper(result, open+1, close, n, s+'(');
        }
        if (close < open) {
            generateParenthesisHelper(result, open, close+1, n, s+')');
        }
    }
    //28
    public int strStr(String haystack, String needle) {

        if (needle == "") return 0;
        int i = haystack.indexOf(needle);
        if (haystack.charAt(i) == needle.charAt(0)) return i;
        return -1;
    }
    //30
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> result = new ArrayList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String word : words){
            int count = map.getOrDefault(word, 0) + 1;
            map.put(word, count);
        }
        int n = s.length(), nums = words.length, size = words[0].length();
        for (int i = 0; i < s.length() -nums * size + 1; i++ ){
            HashMap<String, Integer> tempMap = new HashMap<>();
            int j = 0;
            while (j < nums){
                String tempS = s.substring(i + j * size, i + (j + 1)  * size);
                if (map.containsKey(tempS)){
                    int count = tempMap.getOrDefault(tempS, 0) + 1;
                    if (count <= map.get(tempS)){
                        tempMap.put(tempS, count);
                    }else{
                        break;
                    }
                }else{
                    break;
                }
                j++;
            }
            if (j == nums){
                result.add(i);
            }
        }
        return result;
    }
    //32
    public int longestValidParentheses(String s) {
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i) == ')'){
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '('){
                    stack.pop();
                }else{
                    stack.push(i);
                }
            }else{
                stack.push(i);
            }
        }
        if (stack.isEmpty()) return s.length();
        int end = s.length();
        int start;
        while(!stack.isEmpty()){
            start = stack.pop();
            result = Math.max(result, end - start - 1);
            end = start;
        }

        return result > end ? result : end;
    }

    //38
    public String countAndSay(int n) {
        if (n == 1){
            return "1";
        }
        String s = countAndSay(n-1);
        StringBuilder sb = new StringBuilder();
        List<Character> list = new ArrayList<>();
        HashMap<Character, Integer> map = new HashMap<>();
        int i = 0;
        while (i < s.length()){
            int j = i;
            while (j < s.length() && s.charAt(i) == s.charAt(j)){
                j++;
            }
            Integer temp = (j-i);
            sb.append(temp.toString());
            sb.append(s.charAt(i));
            i = j;
        }

        return sb.toString();
    }
    //43
    public String multiply(String num1, String num2) {
        int[] midResult = new int[num1.length() + num2.length()];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < num1.length() ; i++){
            for (int j = 0; j < num2.length(); j++){
                int temp = (num1.charAt(num1.length()-i-1) - '0') * (num2.charAt(num2.length()-j-1) - '0');
                int temp1 = midResult[i+j] + temp % 10;
                midResult[i+j] = temp1  % 10;
                midResult[i+j+1] = midResult[i+j+1] +  temp / 10 + temp1 /10;
            }
        }
        int i = midResult.length-1;
        while (i >= 0 && midResult[i] == 0){
            i--;
        }
        while ( i >= 0){
            sb.append(midResult[i--]);
        }
        return i == -1 ? "0" : sb.toString();
    }
    //44
    public boolean isMatch2(String s, String p) {
        if (s.length() == 0 && p.length() == 0) return true;
        if (s.length() == 0|| p.length() == 0) return false;
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        dp[0][0] = true;
        for (int i = 0; i < p.length(); i++){
            if ((p.charAt(i) == '?' || p.charAt(i) == '*') && dp[0][i]){
                dp[0][i+1] = true;
            }
        }
        for (int i = 0; i < s.length(); i++){
            for (int j = 0; j < p.length(); j++){
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?'){
                    dp[i+1][j+1] = dp[i][j];
                }else if (p.charAt(j) == '*'){
                    dp[i+1][j+1] = dp[i][j+1] || dp[i][j] || dp[i][j-1];
                }
            }
        }

        return dp[s.length()][p.length()];
    }

    //49
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) return new ArrayList<>();
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs){
            char[] tempChar = new char[26];
            for (char c: str.toCharArray()){
                tempChar[c - 'a']--;
            }
            String tempKey = String.valueOf(tempChar);
            map.computeIfAbsent(tempKey, k -> new ArrayList<>()).add(str);
        }
        return new ArrayList<>(map.values());
    }
    //58
    public int lengthOfLastWord(String s) {
        String[] temp = s.split("\s*");

        return temp[temp.length-1].length();
    }
    //65
    public boolean isNumber(String s) {
        s = s.trim();
        //.
        boolean pointSeen = false;
        //e
        boolean eSeen = false;
        //0-9
        boolean numberSeen = false;
        //0-9 after e
        boolean numberAfterE = false;
        for (int i = 0; i < s.length(); i++){
            if ('0' <= s.charAt(i) && s.charAt(i) <= '9'){
                numberSeen = true;
                numberAfterE = true;
            }else if (s.charAt(i) == '.'){
                if (eSeen || pointSeen){
                    return false;
                }
                pointSeen = true;
            }else if (Character.toLowerCase(s.charAt(i)) == 'e'){
                if (eSeen || !numberSeen){
                    return false;
                }
                eSeen = true;
                numberAfterE = false;
            }else if (s.charAt(i) == '-' || s.charAt(i) == '+'){
                if (i != 0 && s.charAt(i-1) != 'e'){
                    return false;
                }
            }else {
                return false;
            }
        }
        return numberSeen && numberAfterE;
    }
    //67
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1, j = b.length() - 1, carry = 0;
        while (i >= 0 || j >=0){
            int sum = carry;
            if (i >= 0){
                sum += (a.charAt(i--) - '0');
            }
            if (j >= 0){
                sum += (b.charAt(j--) - '0');
            }
            sb.append((sum  % 2));
            carry = sum / 2;
        }
        if (carry != 0) sb.append(carry);
        return sb.reverse().toString();
    }
    //68
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> result = new ArrayList<>();
        int index = 0;
        while (index < words.length){
            int count = words[index].length();
            int last = index + 1;
            while (last < words.length){
                if (words[last].length() + count + 1> maxWidth){
                    break;
                }
                count += (words[last++].length() + 1);
            }
            StringBuilder sb = new StringBuilder();
            int diff = last - index - 1;
            if (last == words.length || diff == 0){
                for (int i = index; i < last; i++){
                    sb.append(words[i] + " ");
                }
                sb.deleteCharAt(sb.length()-1);
                for (int i = sb.length(); i < maxWidth; i++){
                    sb.append(" ");
                }
            }else{
                int spaces = (maxWidth - count) / diff;
                int rest = (maxWidth - count) % diff;
                for (int i = index; i < last; i++){
                    sb.append(words[i]);
                    if (i < last - 1) {
                        for (int j = 0; j <= (spaces + ((i - index) < rest ? 1 : 0)); j++) {
                            sb.append(" ");
                        }
                    }
                }
            }
            result.add(sb.toString());
            index = last;
        }
        return result;
    }
    //71
    public String simplifyPath(String path) {
        Stack<String> stack = new Stack<>();
        HashSet<String> set = new HashSet<>(Arrays.asList("..",".",""));
        for (String s : path.split("/")){
            if (s.equals("..") && !stack.isEmpty()){
                stack.pop();
            }else if (!set.contains(s)){
                stack.push(s);
            }
        }
        String result = "";
        while (!stack.isEmpty()){
            result = '/' + stack.pop() + result;
        }
        return result.length() == 0 ? "/" : result;
    }

    //72
    public int minDistance(String word1, String word2) {
        Integer[][] dp = new Integer[word1.length()+1][word2.length()+1];
        for (int i = 0; i <= word2.length(); i++){
            dp[0][i] = i;
        }
        for (int j = 0; j <= word1.length(); j++){
            dp[j][0] = j;
        }
        for (int i = 0; i < word1.length(); i++){
            for (int j = 0; j < word2.length(); j++){
                if (word1.charAt(i) == word2.charAt(j)){
                    dp[i+1][j+1] = dp[i][j];
                }else{
                    int a = dp[i+1][j];
                    int b = dp[i][j+1];
                    int c = dp[i][j];
                    dp[i+1][j+1] = Math.min(Math.min(a,b),c) + 1;
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    public String minWindow(String s, String t) {
        if (s == null || t.length() > s.length() || s.length() == 0) return "";
        HashMap<Character, Integer> map = new HashMap<>();
        for (Character c : t.toCharArray()){
            int temp = map.getOrDefault(c, 0) + 1;
            map.put(c, temp);
        }
        int count = t.length(), begin = 0, end = 0, maxL = Integer.MAX_VALUE, head = 0;
        while (end < s.length()){
            if (map.containsKey(s.charAt(end))){
                int temp = map.get(s.charAt(end)) - 1;
                if (temp >= 0){
                    count--;
                }
                map.put(s.charAt(end), temp);
            }
            while (count == 0){
                if ((end - begin) < maxL){
                    maxL = end - begin;
                    head = begin;
                }
                if (map.containsKey(s.charAt(begin))){
                    int temp = map.get(s.charAt(begin)) + 1;
                    if (temp > 0){
                        count++;
                    }
                    map.put(s.charAt(begin), temp);
                }
                begin++;
            }
            end++;
        }
        return maxL >= s.length() ? "" : s.substring(head, head + maxL);
    }

    public boolean isScramble(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int len = s1.length();
        boolean [][][] F = new boolean[len][len][len + 1];
        for (int k = 1; k <= len; ++k)
            for (int i = 0; i + k <= len; ++i)
                for (int j = 0; j + k <= len; ++j)
                    if (k == 1)
                        F[i][j][k] = s1.charAt(i) == s2.charAt(j);
                    else for (int q = 1; q < k && !F[i][j][k]; ++q) {
                        F[i][j][k] = (F[i][j][q] && F[i + q][j + q][k - q]) || (F[i][j + k - q][q] && F[i + q][j][k - q]);
                    }
        return F[0][0][len];
    }

    //91
    public int numDecodings(String s) {
        if (s.length() == 0  || s == null) {
            return 0;
        }
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i < s.length(); i++){
            int sing = Integer.valueOf(s.substring(i-1,i));
            int doub = Integer.valueOf(s.substring(i-2,i));
            if (1<=sing && sing<=9){
                dp[i] += dp[i-1];
            }
            if (10 <= doub && doub <=26 ){
                dp[i] += dp[i-2];
            }
        }
        return dp[s.length()];
    }
    //93
    public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<>();
        int n = s.length();
        for (int i = 1 ;i < n-2 &&  i < 4; i++){
            for (int j = i+1;j < n-1 && j < i+4; j++){
                for (int k = j + 1; k < n && k < j + 4 ; k++){
                    String s1 = s.substring(0,i);
                    String s2 = s.substring(i,j);
                    String s3 = s.substring(j,k);
                    String s4 = s.substring(k);
                    if (restoreIpAddressesHelper(s1) && restoreIpAddressesHelper(s2) && restoreIpAddressesHelper(s3) && restoreIpAddressesHelper(s4)){
                        result.add(s1+'.'+s2+'.'+s3+'.'+s4);
                    }
                }
            }
        }
        return result;
    }
    public boolean restoreIpAddressesHelper(String s){
        if (s == null || s.length() == 0 ||s.length() > 3|| (s.charAt(0) == '0' && s.length() >1) || Integer.parseInt(s) > 255){
            return false;
        }
        return true;
    }

    // 97
    public boolean isInterleave(String s1, String s2, String s3) {
        char[] c1 = s1.toCharArray(), c2 = s2.toCharArray(), c3 = s3.toCharArray();
        if (s1.length() + s2.length() != s3.length()) return false;
        return isInterleaveDFS(c1, c2, c3, 0, 0 , 0 , new boolean[s1.length()][s2.length()]);
    }
    public boolean isInterleaveDFS(char[] c1, char[] c2, char[] c3, int i, int j, int k, boolean[][] invalid){
        if (invalid[i][j]) return false;
        if (k == c3.length) return true;
        boolean valid = i < c1.length && c1[i] == c3[k] && isInterleaveDFS(c1, c2, c3, i+1, j, k+1, invalid) ||
                j < c2.length && c2[j] == c3[k] && isInterleaveDFS(c1, c2, c3, i, j+1, k+1, invalid);
        if (!valid) invalid[i][j]= true;
        return valid;
    }
    //115
    public int numDistinct(String s, String t) {
        int[][] dp = new int[s.length()+1][t.length()+1];
        for (int i = 0; i < s.length();i++){
            dp[i][0] = 1;
        }
        for (int i = 0; i < s.length(); i++){
            for (int j = 0; j < t.length(); j++){
                if (s.charAt(i) == t.charAt(j)){
                    dp[i+1][j+1] = dp[i][j] + dp[i][j+1];
                }else{
                    dp[i+1][j+1] = dp[i][j+1];
                }
            }
        }
        return dp[s.length()][t.length()];
    }

    //125
    public boolean isPalindrome(String s) {
        s = s.trim();
        if (s.length() == 0) return true;
        int left = 0, right = s.length()-1;
        while (left < right){
            char leftC = Character.toLowerCase(s.charAt(left));
            char rightC = Character.toLowerCase(s.charAt(right));
            if (!Character.isLetterOrDigit(leftC)){
                left++;
            }else if (!Character.isLetterOrDigit(rightC)){
                right--;
            }else if (leftC != rightC){
                return false;
            }else{
                right--;
                left++;
            }
        }
        return true ;
    }
    //126
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = findLaddersBFS(beginWord, endWord, wordList);
        return result;
    }
    public List<List<String>> findLaddersBFS(String start, String end, List<String> wordList){
        List<String> checked = new ArrayList<>();
        Queue<String> queue = new ArrayDeque<>();
        queue.add(start);
        Queue<List<String>> result = new ArrayDeque<>();
        List<List<String>> endResult = new ArrayList<>();
        result.add(Arrays.asList(start));
        boolean findEnd = false;
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0; i < size; i++){
                List<String> temp  = result.remove();
                String cur = queue.remove();
                checked.add(cur);
                List<String> neigh = findLaddersGetNeighbors(cur, wordList);
                for (String neight : neigh){
                    if (!checked.contains(neight)){
                        queue.add(neight);
                        List<String> tempp = new ArrayList<>(temp);
                        tempp.add(neight);
                        result.add(tempp);
                        if (neight.equals(end)){
                            findEnd = true;
                            endResult.add(tempp);
                        }
                    }
                }
            }
            if (findEnd){
                break;
            }
        }
        return endResult;
    }
    public List<String> findLaddersGetNeighbors(String node, List<String> wordList){
        List<String> result = new ArrayList<>();
        for (String n : wordList){
            int sum = 0;
            for (int i = 0; i < node.length(); i++){
                if (node.charAt(i) != n.charAt(i)){
                    sum++;
                }
            }
            if (sum == 1){
                result.add(n);
            }
        }
        return result;
    }
    //127
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        HashSet<String> unchecked = new HashSet<>(wordList);
        Queue<String> queue = new ArrayDeque<>();
        queue.add(beginWord);
        boolean findEnd = false;
        int result = 1;

        while (!queue.isEmpty()){
            int size = queue.size();
            result++;
            for (int i = 0; i < size; i++){
                String cur = queue.remove();
                unchecked.remove(cur);
                List<String> neigh = ladderLengthHelper(cur, unchecked);
                for (String neight : neigh){
                    if (unchecked.contains(neight)){
                        queue.add(neight);
                        if (neight.equals(endWord)){
                            findEnd = true;
                        }
                    }
                }
            }
            if (findEnd){
                break;
            }
        }
        return findEnd ? result : 0;
    }
    public List<String> ladderLengthHelper(String node, HashSet<String> unchecked) {
        List<String> result = new ArrayList<>();
        char chs[] = node.toCharArray();

        for (char ch ='a'; ch <= 'z'; ch++) {
            for (int i = 0; i < chs.length; i++) {
                if (chs[i] == ch) continue;
                char old_ch = chs[i];
                chs[i] = ch;
                if (unchecked.contains(String.valueOf(chs))) {
                    result.add(String.valueOf(chs));
                }
                chs[i] = old_ch;
            }

        }
        return result;
    }
    //131
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        partitionBT(s, 0, new ArrayList<>(), result);
        return result;
    }
    public void partitionBT(String s, int j, List<String> curL, List<List<String>> result){
        if (j > s.length() -1){
            result.add(new ArrayList<>(curL));
            return;
        }
        int size = curL.size();
        for (int i = 0; i < s.length() - j; i++){
            if (partitionIsPalindrome(s, j, j+i)){
                curL.add(s.substring(j, j+i+1));
                partitionBT(s, j+i+1, curL, result);
            }
            while (curL.size() > size){
                curL.remove(curL.size()-1);
            }
        }
    }
    public boolean partitionIsPalindrome(String s, int left , int right){
        while ( left < right){
            if (s.charAt(left++) != s.charAt(right--)){
                return false;
            }
        }
        return true;
    }
    //132
    public int minCut(String s) {
        int n = s.length();
        int[] cut = new int[n];
        boolean[][] pal = new boolean[n][n];
        for (int i = 0; i < n; i++){
            int min = i;
            for (int j = 0; j <= i; j++){
                if (s.charAt(i) == s.charAt(j) && ((j+1 > i-1) || pal[j+1][i-1])){
                    pal[j][i] = true;
                    min = j == 0 ? 0 : Math.min(min, cut[j-1] + 1);
                }
            }
            cut[i] = min;
        }
        return cut[n-1];
    }
    //139
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for (int i = 0; i <= s.length(); i++){
            for (int j = 0; j < i; j++){
                if (dp[j] && wordDict.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
    //140
    public List<String> wordBreak2(String s, List<String> wordDict) {
        List<String> result = new ArrayList<>();
        wordBreak2BT(s, 0, wordDict, result, new ArrayList<>());
        return result;
    }
    public void wordBreak2BT(String s, int j, List<String> wordDict, List<String> result, List<String> midString){
        if (j >= s.length()){
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < midString.size(); i++){
                sb.append(midString.get(i));
                sb.append(" ");
            }
            result.add(sb.toString().trim());
            return;
        }
        int size = midString.size();
        for (int i = 1; i <= s.length() - j; i++){
            if (wordDict.contains(s.substring(j, j+i))){
                midString.add(s.substring(j, j+i));
                wordBreak2BT(s, j+i, wordDict, result, midString);
            }
            while (midString.size() > size){
                midString.remove(midString.size()-1);
            }
        }
    }
    //151
    public String reverseWords(String s) {
        s = s.trim();
        String[] temp = s.split("\s{1,100}");
        StringBuilder sb = new StringBuilder();
        for (int i = temp.length-1; i>=0; i--){
            sb.append(temp[i]);
            sb.append(" ");
        }
        return sb.toString().trim();
    }
    public int minCutBT(String s, int j, int cut, int[][] dp){
        if (j == s.length()-1){
            return cut;
        }else if (j >= s.length()){
            return cut-1;
        }
        int minSum = Integer.MAX_VALUE;
        for (int i = 0; i < s.length() - j; i++){
            if (dp[j][j+i] == 0){
                dp[j][j+i] = partitionIsPalindrome(s, j, j+i) ? 1 : -1;
            }
            if (dp[j][j+i] == 1){
                int temp = minCutBT(s, j+i+1, cut+1, dp);
                minSum = Math.min(temp, minSum);
            }
        }
        return minSum;
    }
    //165
    public int compareVersion(String version1, String version2) {
        List<String> version1C = Arrays.asList(version1.split("\\."));
        List<String> version2C = Arrays.asList(version2.split("\\."));
        version1C = new ArrayList<>(version1C);
        version2C = new ArrayList<>(version2C);
        while (version1C.size() != version2C.size()){
            if (version1C.size() < version2C.size()){
                version1C.add("0");
            }else{
                version2C.add("0");
            }
        }
        int n = version1C.size();
        for (int i = 0; i < n; i++){
            int temp1 = Integer.valueOf(version1C.get(i));
            int temp2 = Integer.valueOf(version2C.get(i));
            if (temp1 > temp2){
                return 1;
            }else if (temp1 < temp2){
                return -1;
            }
        }
        return 0;
    }
    public static void main(String[] args) {
        Heap solution = new Heap();
        solution.minCut("leet");
        System.out.println(Integer.valueOf("001"));
        List<String> temp = Arrays.asList("1.01.01".split("\\."));
        List tempp = new ArrayList(temp);
        tempp.add("0");
        System.out.println(temp);
        solution.compareVersion("1.01.01", "1.001");




//        System.out.println(99*99);
    }

}
