import edu.princeton.cs.algs4.Quick3string;

import java.nio.charset.StandardCharsets;
import java.util.*;

public class Solutions2 {
//    public class Node {
//        public int val;
//        public Node left;
//        public Node right;
//        public Node next;
//
//        public Node() {}
//
//        public Node(int _val) {
//            val = _val;
//        }
//
//        public Node(int _val, Node _left, Node _right, Node _next) {
//            val = _val;
//            left = _left;
//            right = _right;
//            next = _next;
//        }
//    };
//    public Node connect(Node root) {
//        Node level = root;
//        while (level != null){
//            Node cur = level;
//            while (cur != null){
//                if (cur.left != null){
//                    cur.left.next = cur.right;
//                }
//                if (cur.right != null && cur.next != null){
//                    cur.right.next = cur.next.left;
//                }
//                cur = cur.next;
//            }
//            level = level.left;
//        }
//        return root;
//    }
//    public Node connect2(Node root) {
//        Node level = root;
//        Node prev = null;
//        Node head = null;
//        while (level != null){
//            Node cur = level;
//            while (cur != null){
//                if (cur.left != null){
//                    if (prev != null){
//                        prev.next = cur.left;
//                    }
//                    else{
//                        head = cur.left;
//                    }
//                    prev = cur.left;
//
//                }
//                if (cur.right != null){
//                    if (prev != null){
//                        prev.next = cur.right;
//                    }else{
//                        head = cur.right;
//                    }
//                    prev = cur.right;
//                }
//                cur = cur.next;
//            }
//            prev = null;
//            level = head;
//            head = null;
//        }
//        return root;
//    }
public class Node {
    public int val;
    public List<Node> neighbors;
    public Node() {
        val = 0;
        neighbors = new ArrayList<Node>();
    }

    public Node(int _val) {
        val = _val;
        neighbors = new ArrayList<Node>();
    }
    public Node(int _val, ArrayList<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }}


    public Node cloneGraph(Node node) {
        HashMap<Integer, Node> a= new HashMap<>();
        if (node == null) return node;
        cloneGraphHelper(node, a);
        return a.get(0);
    }
    public void cloneGraphHelper(Node node, HashMap<Integer, Node> a){
        if (a.containsKey(node.val)) return;
        Node clone = new Node(node.val);
        a.put(node.val, clone);
        for (Node neigh: node.neighbors){
            cloneGraphHelper(neigh, a);
            clone.neighbors.add(a.get(neigh.val));
        }
    }
    //207
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        int[] source = new int[numCourses];
        for (int i = 0; i < prerequisites.length; i++){
            int a = prerequisites[i][1];
            int b = prerequisites[i][0];
            List<Integer> temp = adj.get((Integer) a);
            if (temp == null) temp = new ArrayList<>();
            temp.add(b);
            source[b]++;
            adj.put(a, temp);
        }
        int count = 0;
        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++){
            if (source[i] == 0) {
                queue.add(i);
                count++;
            }
        }

        while (!queue.isEmpty()){
            int temp = queue.remove();
            if (adj.get(temp)!= null){
                List<Integer> nodes = adj.get(temp);
                for (Integer node : nodes){
                    if (--source[node] ==0){
                        queue.add(node);
                        count++;
                    }
                }
            }

        }
        return count == numCourses;
    }
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] result = new int[numCourses];
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        int[] indegree = new int[numCourses];
        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < prerequisites.length; i++){
            int a = prerequisites[i][1];
            int b = prerequisites[i][0];
            indegree[b]++;
            List<Integer> temp = adj.get(a);
            if (temp==null) temp = new ArrayList<>();
            temp.add(b);
            adj.put(a, temp);
        }
        for (int i = 0; i <numCourses; i++ ){
            if (indegree[i] == 0){
                queue.add(i);
            }
        }
        int point = 0;
        while (!queue.isEmpty()){
            Integer node = queue.remove();
            result[point++] = node;
            if (adj.get(node)!=null){
                List<Integer> temp = adj.get(node);
                for (Integer neigh : temp){
                    if (--indegree[neigh] == 0){
                        queue.add(neigh);
                    }
                }
            }
        }
        if (point != numCourses){
            return new int[0];
        }
        return result;
    }
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> leaves = new ArrayList<>();
        if (n==1){
            leaves.add(0);
            return leaves;
        }
        List<HashSet<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) adj.add(new HashSet<>());
        for (int i = 0; i < edges.length; i++){
            int a = edges[i][0];
            int b = edges[i][1];
            adj.get(a).add(b);
            adj.get(b).add(a);
        }
        for (int i = 0; i < n; i++){
            if (adj.get(i).size()== 1) leaves.add(i);
        }

        while (n>2){
            List<Integer> tempLeaves = new ArrayList<>();
            n -= leaves.size();
            for (Integer leave : leaves){
                int neighb = adj.get(leave).iterator().next();
                adj.get(neighb).remove(leave);
                if (adj.get(neighb).size() == 1){
                    tempLeaves.add(neighb);
                }
            }
            leaves = tempLeaves;
        }
        return leaves;
    }

    public int longestIncreasingPath(int[][] matrix) {
        int n = matrix.length * matrix[0].length;
        HashMap<Integer, HashSet<Integer>> adj = new HashMap<>();
        for (int i = 0; i < n; i++){
            adj.put(i, new HashSet<>());
        }

        boolean[] root = new boolean[n];
        for (int i = 0; i < n; i++) root[i] = true;

        for (int j = 0; j <matrix.length; j++){
            for (int i = 0; i < matrix[0].length; i++){
                int ID = i + matrix[0].length*j;
                int cur = matrix[j][i];
                if (j >= 1) {
                    int up = matrix[j-1][i];
                    if (up>cur) {
                        adj.get(ID).add(ID - matrix[0].length);
                        root[ID - matrix[0].length] = false;
                    }
                }
                if (j< matrix.length-1){
                    int down = matrix[j+1][i];
                    if (down>cur) {
                        adj.get(ID).add(ID + matrix[0].length);
                        root[ID + matrix[0].length] = false;
                    }
                }
                if (i>=1){
                    int left = matrix[j][i-1];
                    if (left > cur) {
                        adj.get(ID).add(ID - 1);
                        root[ID - 1] = false;
                    }
                }
                if (i< matrix[0].length-1){
                    int right = matrix[j][i+1];
                    if (right > cur) {
                        adj.get(ID).add(ID + 1);
                        root[ID + 1] = false;
                    }
                }
            }
        }

        int max = 0;
        Integer[] pathSum = new Integer[n];
        List<Integer> roots = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (root[i] ){
                roots.add(i);
            }
            pathSum[i] = -1;
        }

        for (Integer i : roots){
            int subPath;
            if (pathSum[i] != -1){
                subPath = pathSum[i];
            }
            else {
                subPath = longestIncreasingPathDFS(adj, i, pathSum);
            }
            if (subPath > max) max = subPath;
        }

        return max;
    }
    public int longestIncreasingPathDFS(HashMap<Integer, HashSet<Integer>> adj, int i, Integer[] pathSum){
        if (pathSum[i] != -1){
            return pathSum[i];
        }
        if (adj.get(i).size() == 0) {
            pathSum[i] = 1;
            return 1;
        }
        int max = 0;
        for (int j : adj.get(i)){
            int subPath;
            if (pathSum[j] != -1){
                subPath = pathSum[j];
            }else{
               subPath = longestIncreasingPathDFS(adj, j, pathSum);
            }
            if (subPath > max){
                max = subPath;
            }
        }
        pathSum[i] = max + 1;
        return max + 1;
    }
    //332


    public List<String> findItinerary(List<List<String>> tickets){
        HashMap<String, PriorityQueue<String>> adj = new HashMap<>();
        List<String> result = new ArrayList<>();
        for (List<String> ss : tickets){
            adj.computeIfAbsent(ss.get(0), k -> new PriorityQueue<>()).add(ss.get(1));
        }
        findItineraryHelper("JFK", adj, result);
        return result;
    }
    public void findItineraryHelper(String cur, HashMap<String, PriorityQueue<String>> adj, List<String> result){
        while (adj.containsKey(cur) && adj.get(cur).size() != 0){
            findItineraryHelper(adj.get(cur).poll(), adj, result);
        }
        result.add(0, cur);
    }
//    public List<String> findItinerary(List<List<String>> tickets) {
//        List<List<String>> result = new ArrayList<>();
//        HashMap<String, List<String>> adj = createADJ(tickets);
//        List<String> path = new ArrayList<>();
//        findItinerayDFS(adj, "JFK",path, result);
//        return result.get(0);
//    }
//    public HashMap<String, List<String>> createADJ(List<List<String>> tickets){
//        HashMap<String, List<String>> adj = new HashMap<>();
//        for (List<String> ticket : tickets){
//            List<String> des = adj.get(ticket.get(0));
//            if (des == null) des = new ArrayList<>();
//            des.add(ticket.get(1));
//            adj.put(ticket.get(0), des);
//        }
//        return adj;
//    }
//    public void findItinerayDFS(HashMap<String, List<String>> adj, String key,List<String> path, List<List<String>> result){
//        path.add(key);
//        int size = path.size();
//        boolean clear = true;
//        for (String s : adj.keySet()){
//            if (adj.get(s).size() != 0){
//                clear = false;
//            }
//        }
//        if ((adj.get(key) == null || adj.get(key).size() == 0) && clear) {
//            if (result.size() == 0) {
//                result.add(new ArrayList<>(path));
//            }else{
//                List<String> tempPath = result.get(0);
//                String pathString = String.join("", path);
//                String tempString = String.join("" , tempPath);
//                if (pathString.compareTo(tempString) < 0){
//                    result.remove(0);
//                    result.add(new ArrayList<>(path));
//                    return;
//                }
//            }
//
//            return;
//        }else if (adj.get(key)!= null) {
//            List<String> temp = new ArrayList<>(adj.get(key));
//            for (String next : temp) {
//                adj.get(key).remove(next);
//                findItinerayDFS(adj, next, path, result);
//                adj.get(key).add(next);
//            }
//            }
//        int tempSize = path.size();
//        for (int i = path.size() - 1; i >= size - 1; i--) {
//            path.remove(i);
//        }
//    }
    //399
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        HashMap<String, HashMap<String, Double>> adj = new HashMap<>();
        Union union = new Union();
        for (int i = 0; i < equations.size(); i++){
            List<String> equation = equations.get(i);
            String a1 = equation.get(0);
            String a2 = equation.get(1);
            union.connect(a1, a2);
            double val = values[i];
            adj.computeIfAbsent(a1, k->new HashMap<>()).put(a2, val);
            adj.computeIfAbsent(a2, k->new HashMap<>()).put(a1, 1/val);
        }
        double[] result = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++){
            List<String> query = queries.get(i);
            double val = 0;
            if (union.connected(query.get(0), query.get(1))) {
                val = calcEquationHelper(query.get(0), query.get(1), adj);
            }
            if (val == 0) val = -1;
            result[i] = val;
        }
        return result;
    }
    public static class Union{
        private final HashMap<String, String> parent;
        public Union(){
            parent = new HashMap<>();
        }
        public void connect(String a, String b){
            parent.putIfAbsent(a, a);
            parent.putIfAbsent(b, b);
            String roota = root(a);
            String rootb = root(a);
            parent.put(rootb, roota);
        }
        public boolean connected(String a, String b){
            if (!parent.containsKey(a) || !parent.containsKey(b)){
                return false;
            }
            String roota = root(a);
            String rootb = root(b);
            if (roota.equals(rootb)) return true;
            return false;
        }
        public String root(String a){
            String tempP = parent.get(a);
            while (!tempP.equals(parent.get(tempP))){
                tempP = parent.get(tempP);
            }
            return tempP;
        }
    }

    public double calcEquationHelper(String start, String end, HashMap<String, HashMap<String, Double>> adj){
        if (adj.containsKey(start) && start.equals(end)) return 1;
        if (!adj.containsKey(start) || adj.get(start).size() == 0) return 0;
        HashMap<String, Double> tempMap = new HashMap<>(adj.get(start));
        for (String key : tempMap.keySet()){
            if (key.equals(end)){
                return adj.get(start).get(key);
            }
            double tempVal = adj.get(start).get(key);
            adj.get(start).remove(key);
            double val = tempVal * calcEquationHelper(key, end, adj);
            adj.get(start).put(key, tempVal);
            if (val != 0) {
                return val;
            }
        };
        return 0;
    }
//    public class edgeNode{
//        private edgeNode next;
//        private double weight;
//        private String val;
//        public edgeNode(){
//
//        }
//        public edgeNode(String val, double weight){
//            this.val = val;
//            this.weight = weight;
//            this.next = null;
//        }
//        public edgeNode(String val, double weight, edgeNode node){
//            this.val = val;
//            this.weight = weight;
//            this.next = node;
//        }
//    }
    //547
    public int findCircleNum(int[][] isConnected) {
        int[] parent = new int[isConnected.length];
        int count = isConnected.length;
        for (int i = 0; i < parent.length; i++){
            parent[i] = i;
        }
        for (int i = 0; i < parent.length; i++){
            for (int j = 0; j < i; j++){
                if (isConnected[i][j] == 1){
                    int a = findCircleNumHelper(parent, i);
                    int b = findCircleNumHelper(parent, j);
                    parent[a] = b;
                }
            }
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < parent.length; i++){
            set.add(findCircleNumHelper(parent, i));
        }
        return set.size();
    }
    public int findCircleNumHelper(int[] parent, int a){
        while (a!=parent[a]){
            a = parent[a];
        }
        return a;
    }

    //684
    public int[] findRedundantConnection(int[][] edges) {
        int[] parent = new int[edges.length];
        for (int i = 0; i < parent.length; i++) parent[i] = i;
        for (int i = 0; i < edges.length;i++){
            int v1 = edges[i][0];
            int v2 = edges[i][1];
            int root1 = findRedundantConnection(parent, v1);
            int root2 = findRedundantConnection(parent, v2);
            if ( root1 == root2){
                return edges[i];
            }else{
                parent[root1] = root2;
            }
        }
        return null;
    }
    public int findRedundantConnection(int[] parent, int i){
        while (i != parent[i]){
            i = parent[i];
        }
        return i;
    }
    //685
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int[] parent = new int[edges.length+1];
        int[] multiP = null;
        int[] sameP = null;
        for (int i = 0; i < parent.length; i++) parent[i] = i;
        for (int i = 0; i < edges.length; i++){
            int root1 = findRedundantDirectedConnectionHelper(parent, edges[i][0]);
            int root2 = findRedundantDirectedConnectionHelper(parent, edges[i][1]);
            if (root1 == root2) sameP = edges[i];
            else if (root2 != edges[i][1]){
                multiP = edges[i];
            }else{
                parent[root2] = root1;
            }
        }
        if (sameP != null && multiP == null)  return sameP;
        if (multiP != null && sameP == null) return multiP;
        if (multiP != null && sameP != null) {
            for (int[] edge : edges) {
                if (edge[1] == multiP[1]){
                    return edge;
                }
            }
        }
        return null;
    }
    public int findRedundantDirectedConnectionHelper(int[] parent, int p){
        while (p != parent[p]){
            p = parent[p];
        }
        return p;
    }

    //734
    public int networkDelayTime(int[][] times, int n, int k) {
        HashMap<Integer, HashMap<Integer, Integer>> adj  = new HashMap<>();
        for (int i = 0; i < times.length; i++){
            int[] temp = times[i];
            adj.computeIfAbsent(temp[0], Key -> new HashMap<>()).put(temp[1], temp[2]);
        }
        PriorityQueue<Integer[]> queue = new PriorityQueue<>((a,b) -> (a[0]-b[0]));
        queue.add(new Integer[]{0,k});
        Integer result = 0;
        boolean[] visited = new boolean[n+1];
        while (!queue.isEmpty()){
            Integer[] cur = queue.remove();
            Integer curDist = cur[0];
            Integer curNode = cur[1];
            if (!visited[curNode]){
                visited[curNode] = true;
                n--;
                result = curDist;
                if (adj.containsKey(curNode)) {
                    for (Integer Key : adj.get(curNode).keySet()) {
                        Integer tempDist = adj.get(curNode).get(Key);
                        queue.add(new Integer[]{curDist + tempDist, Key});
                    }
                }
            }


        }
        return n == 0 ? result: -1;
    }

    //785
    public boolean isBipartite(int[][] graph) {
        boolean[] checked = new boolean[graph.length];
        String[] color = new String[graph.length];
        for (int i = 0; i < graph.length; i++){
            color[i] = "UNCOLORED";
        }
        for (int cur = 0; cur < graph.length; cur++){
            if (!checked[cur]){
                color[cur] = "WHITE";
                checked[cur] = true;
                Queue<Integer> queue = new ArrayDeque<>();
                queue.add(cur);
                while (!queue.isEmpty()){
                    int tempCur = queue.remove();
                    for (int next : graph[tempCur]){
                        if (color[tempCur] == color[next]){
                            return false;
                        }else{
                            color[next] = isBipartiteHelper(color[tempCur]);
                        }
                        if (!checked[next]){
                            queue.add(next);
                            checked[next] = true;
                        }
                    }
                }
            }
        }
        return true;
    }
    public String isBipartiteHelper(String color){
        if (color == "WHITE"){
            return "BLACK";
        }
        return "WHITE";
    }

    //787
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        HashMap<Integer, HashMap<Integer, Integer>> graph = new HashMap<>();
        for (int i = 0; i < flights.length; i++){
            int cur = flights[i][0];
            int dest = flights[i][1];
            int price = flights[i][2];
            graph.computeIfAbsent(cur, Key -> new HashMap<>()).put(dest, price);
        }
        int[] cost = new int[n];
        int[] stop = new int[n];
        for (int i = 0; i < n; i ++){
            cost[i] = Integer.MAX_VALUE;
            stop[i] = k+1;
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[0]-b[0]);
        queue.add(new int[]{0, src, k+1});
        while (!queue.isEmpty()){
            int[] temp = queue.remove();
            int K = temp[2];
            int price = temp[0];
            int cur = temp[1];
            if (cur == dst) return price;
            if (K > 0 && graph.containsKey(cur)){
                for (Integer key : graph.get(cur).keySet()){
                    int tempPrice = price + graph.get(cur).get(key);
                    if (tempPrice < cost[key]) {
                        cost[key] = tempPrice;
                        stop[key] = k - K;
                        queue.add(new int[]{tempPrice, key, K - 1});
                    }else if ((k-K )< stop[key]){
                        queue.add(new int[]{tempPrice, key, K - 1});
                    }
                }
            }
        }
        return -1;
    }
    //797
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> result= new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        path.add(0);
        allPathsSourceTargetHelper(graph, path, result, 0);
        return result;
    }
    public void allPathsSourceTargetHelper(int[][] graph, List<Integer> path, List<List<Integer>> result, int key){
        if (key == graph.length-1){
            result.add(new ArrayList<>(path));
            return;
        }
        for (Integer next : graph[key]){
            path.add(next);
            allPathsSourceTargetHelper(graph, path, result, next);
            path.remove(path.size()-1);
        }
    }
    //802
    public List<Integer> eventualSafeNodes(int[][] graph) {
        List<Integer> result = new ArrayList<>();
        int[] dp = new int[graph.length];
        Arrays.fill(dp, -1);
        for (int i = 0; i < graph.length; i++){
            if (dp[i] == 1) {
                result.add(i);
                continue;
            }
            if (graph[i].length == 0){
                result.add(i);
            }else{
                boolean[] checked = new boolean[graph.length];
                checked[i] = true;
                boolean temp = eventualSafeNodesHelper(graph, i, checked, dp);
                if (temp){
                    result.add(i);
                }
            }
        }
        return result;
    }
    public boolean eventualSafeNodesHelper(int[][] graph, int key, boolean[] checked, int[] dp){
        if (dp[key] == 1) return true;
        if (graph[key].length == 0){
            dp[key] = 1;
            return true;
        }
        for (int next : graph[key]){
            if (checked[next]){
                dp[key] = 0;
                return false;
            }else{
                checked[next] = true;
                boolean temp = eventualSafeNodesHelper(graph, next, checked, dp);
                if (!temp){
                    dp[key] = 0;
                    checked[next] = false;
                    return false;
                }
                checked[next] = false;
            }
        }
        dp[key] = 1;
        return true;
    }
    //834
    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        for (int[] edge: edges){
            adj.computeIfAbsent(edge[1], k->new ArrayList<>()).add(edge[0]);
            adj.computeIfAbsent(edge[0], k->new ArrayList<>()).add(edge[1]);
        }
        for (int i = 0; i < n; i++){
            adj.computeIfAbsent(i, k->new ArrayList<>());
        }
        int[] result = new int[n];
        int[] count = new int[n];
        boolean[] checked = new boolean[n];
        checked[0] = true;
        sumOfDistancesInTreeDFSPost(0, adj, count, result, checked);
        checked = new boolean[n];
        checked[0] = true;
        sumOfDistancesInTreeDFSPre(0, adj, count, result, checked);
        return result;
    }

    public void sumOfDistancesInTreeDFSPost(int cur, HashMap<Integer, List<Integer>> adj, int[] count, int[] res, boolean[] checked){
        checked[cur] = true;

        for (int next : adj.get(cur)){
            if (!checked[next]) {
                sumOfDistancesInTreeDFSPost(next, adj, count, res, checked);
                count[cur] += count[next];
                res[cur] += res[next] + count[next];
            }
        }
        count[cur]++;
    }
    public void sumOfDistancesInTreeDFSPre(int cur, HashMap<Integer, List<Integer>> adj, int[] count, int[] res, boolean[] checked){
        checked[cur] = true;
        for (int next : adj.get(cur)){
            if (!checked[next]) {
                res[next] = res[cur] - count[next] + adj.size() - count[next];
                sumOfDistancesInTreeDFSPre(next, adj, count, res, checked);
            }
        }
    }
    // 841
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        boolean[] checked = new boolean[n];
        int sum = 0;

        canVisitAllRoomsDFS(rooms, 0, checked);
        for (int i = 0; i < n; i++){
            if (checked[i]) sum++;
        }
        return sum == n;
    }
    public void canVisitAllRoomsDFS(List<List<Integer>> rooms, int cur, boolean[] checked){
        checked[cur] = true;
        for (Integer next : rooms.get(cur)){
            if (!checked[next]){
                canVisitAllRoomsDFS(rooms, next, checked);
            }
        }
    }

    //886
    public boolean possibleBipartition(int n, int[][] dislikes) {
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        for (int[] dislike : dislikes){
            adj.computeIfAbsent(dislike[1], k->new ArrayList<>()).add(dislike[0]);
            adj.computeIfAbsent(dislike[0], k->new ArrayList<>()).add(dislike[1]);
        }
        boolean[] checked = new boolean[n+1];
        String[] color = new String[n+1];
        for (int i = 1; i <= n; i++){
            if (!checked[i]) {
                color[i] = "WHITE";
                boolean temp = possibleBipartitionHelper(i, adj, checked, color);
                if (!temp) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean possibleBipartitionHelper(int cur, HashMap<Integer, List<Integer>> adj, boolean[] checked, String[] color){
        checked[cur] = true;
        if (adj.containsKey(cur)) {
            for (Integer next : adj.get(cur)) {
                if (!checked[next]) {
                    if (color[cur] == color[next]) return false;
                    if (color[cur] == "WHITE") color[next] = "BLACK";
                    if (color[cur] == "BLACK") color[next] = "WHITE";
                    boolean temp = possibleBipartitionHelper(next, adj, checked, color);
                    if (!temp) return false;
                }
            }
        }
        return true;
    }
    //997
    public int findJudge(int n, int[][] trust) {
        int[] indegree = new int[n+1];
        int[] trst = new int[n+1];
        for (int i = 0; i < trust.length; i++){
            indegree[trust[i][1]]++;
            trst[trust[i][0]]++;
        }
        for (int i = 1; i <= n; i++){
            if (indegree[i] == n-1 && trst[i] == 0){
                return i;
            }
        }
        return -1;
    }
    //1319
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n-1) return -1;
        int[] parent = new int[n];
        for (int i = 0; i < n; i++){
            parent[i] = i;
        }
        for (int i = 0; i < connections.length; i++){
            int root1 = makeConnectedRoot(parent,connections[i][0]);
            int root2 = makeConnectedRoot(parent,connections[i][1]);
            if (root1 != root2){
                parent[root2] = root1;
            }
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i++){
            int tempRoot = makeConnectedRoot(parent, i);
            set.add(tempRoot);
        }
        return set.size()-1;
    }
    public int makeConnectedRoot(int[] parent, int i) {
        while (i != parent[i]){
            i = parent[i];
        }
        return i;
    }
    //1466
    public int minReorder(int n, int[][] connections) {
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        for (int[] connection : connections){
            adj.computeIfAbsent(connection[0], k->new ArrayList<>()).add(connection[1]);
            adj.computeIfAbsent(connection[1], k->new ArrayList<>()).add(-connection[0]);
        }
        boolean[] checked = new boolean[n];

        return minReorderDFS(adj, 0, checked);
    }

    public int minReorderDFS(HashMap<Integer, List<Integer>> adj, int cur, boolean[] checked){
        int change = 0;
        checked[cur] = true;
        if (adj.containsKey(cur)){
            for (int next : adj.get(cur)){
                if (!checked[Math.abs(next)]){
                    change += minReorderDFS(adj, Math.abs(next), checked) + (next > 0 ? 1 : 0);
                }
            }
        }
        return change;
    }
    //1557
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        int[] indegree = new int[n];
        for (List<Integer> edge : edges){
            adj.computeIfAbsent(edge.get(0), k->new ArrayList<>()).add(edge.get(1));
            indegree[edge.get(1)]++;
        }
        List<Integer> result = new ArrayList<>();
        boolean[] checked = new boolean[n];
        for (int i = 0; i < n; i++){
            if (indegree[i] == 0){
                result.add(i);
            }
        }
        return result;
    }

    //1791
    public int findCenter(int[][] edges) {
        int[] indegree = new int[edges.length+2];
        for (int[] edge : edges){
            indegree[edge[0]]++;
            indegree[edge[1]]++;
        }
        for (int i = 1; i < indegree.length;i++){
            if (indegree[i] == edges.length){
                return i;
            }
        }

        return 1;
    }

    //1971
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++){
            parent[i] = i;
        }
        for (int[] edge : edges){
            int roota = validPathHelper(parent, edge[0]);
            int rootb = validPathHelper(parent, edge[1]);
            if (roota != rootb){
                parent[rootb] = roota;
            }
        }
        return validPathHelper(parent, source) == validPathHelper(parent, destination);
    }
    public int validPathHelper(int[] parent, int key){
        while (parent[key] != key){
            key = parent[key];
        }
        return key;
    }
    //241
    public static List<Integer> diffWaysToCompute(String expression) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < expression.length(); i++){
            if (expression.charAt(i) == '*' || expression.charAt(i) == '+' || expression.charAt(i) == '-'){
                List<Integer> part1 = diffWaysToCompute(expression.substring(0,i));
                List<Integer> part2 = diffWaysToCompute(expression.substring(i+1));
                for (int p1 : part1){
                    for (int p2 : part2){
                        switch (expression.charAt(i)){
                            case '*':
                                result.add(p1*p2);
                                break;
                            case '+':
                                result.add(p1+p2);
                                break;
                            case '-':
                                result.add(p1-p2);
                                break;
                        }
                    }
                }
            }
        }
        if (result.isEmpty()){
            result.add(Integer.parseInt(expression));
        }
        return result;
    }

    public static void main(String[] args) {
        List<Integer> result = diffWaysToCompute("2*3-4*5");
        int a = 2;
        System.out.println(--a == 1);
        Solutions2 solutions2 = new Solutions2();
//        int[][] aa = new int[4][2];
//        aa[0] = new int[]{1,4};
//        aa[1] = new int[]{2,4};
//        aa[2] = new int[]{3,1};
//        aa[3] = new int[]{3,2};
//        solutions2.canFinish(5, aa);

//        int[][] aa = new int[1][2];
//        aa[0] = new int[]{1,2};
//        solutions2.longestIncreasingPath(aa);
//        System.out.println("ABF".compareTo("ABA"));
//        HashMap<String, List<String>> adj = new HashMap<>();
//        List<String> temp = new ArrayList<>();
//        temp.add("SFO");
//        temp.add("LHR");
//        adj.put("JFK", temp);
//        temp = new ArrayList<>(adj.get("JFK"));
//        for (String key : temp){
//            adj.get("JFK").remove(key);
//            System.out.println(key);
//            adj.get("JFK").add(key);
//        }
//        List<List<String>> temp = new ArrayList<>();
//        List<String> tempp = new ArrayList<>();
//        tempp.add("JFK");
//        tempp.add("SFO");
//        temp.add(tempp);
//        tempp = new ArrayList<>();
//        tempp.add("JFK");
//        tempp.add("ATL");
//        temp.add(tempp);
//        tempp = new ArrayList<>();
//        tempp.add("SFO");
//        tempp.add("ATL");
//        temp.add(tempp);
//        tempp = new ArrayList<>();
//        tempp.add("ATL");
//        tempp.add("JFK");
//        temp.add(tempp);
//        tempp = new ArrayList<>();
//        tempp.add("ATL");
//        tempp.add("SFO");
//        temp.add(tempp);
//        HashMap<String, PriorityQueue<String>> adj = new HashMap<>();
//        for (List<String> ss : temp){
//            adj.computeIfAbsent(ss.get(0), k -> new PriorityQueue<>()).add(ss.get(1));
//        }
//
//        tempp = solutions2.findItinerary(temp);
//        System.out.println(tempp);
        String a1 = "AAA";
        String a2 = "BBB";
        System.out.println(a1==a2);
        System.out.println(a1.equals(a2));
        Solutions2.Union union = new Union();
        union.connect(a1, a2);
        union.connected(a1,a2);
        boolean[] checked = new boolean[2];
        System.out.println(checked[0]);

    }
}

